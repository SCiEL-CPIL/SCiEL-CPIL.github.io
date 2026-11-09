# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hex_edge_normals(angle_deg):
    """Return 6 unit edge normals for a hexagon with given angle."""
    a = np.radians(angle_deg)
    normal_angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])


def _min_outer_side(centers, angles, rot_deg):
    """
    Compute minimum side length s of outer hexagon (rotated by rot_deg) containing all inner hexagons.
    Returns (s, ox, oy).
    """
    sqrt3 = np.sqrt(3)
    rot = np.radians(rot_deg)
    
    n_inner = len(centers)
    all_verts = np.zeros((n_inner * 6, 2))
    for i in range(n_inner):
        cx, cy = centers[i]
        all_verts[6*i:6*(i+1)] = _hex_vertices(cx, cy, 1.0, angles[i])
    
    A = np.zeros((6, 2))
    b = np.zeros(6)
    
    for i in range(6):
        n_angle = rot + np.radians(60 * i + 30)
        nx = np.cos(n_angle)
        ny = np.sin(n_angle)
        A[i] = [nx, ny]
        
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        p_min = projections.min()
        p_max = projections.max()
        
        b[i] = (p_min + p_max) / 2
    
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    ox, oy = c
    
    s_final = 0.0
    center_proj = ox * A[:, 0] + oy * A[:, 1]
    for i in range(6):
        nx, ny = A[i]
        d_proj = all_verts[:, 0] * nx + all_verts[:, 1] * ny - center_proj[i]
        s_final = max(s_final, np.abs(d_proj).max() * 2 / sqrt3)
    
    return s_final, ox, oy


def _separation_check(c1, a1, c2, a2, eps=1e-6):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated.
    """
    n1 = np.radians(a1)
    normals1 = np.column_stack([np.cos(n1 + np.radians(60 * np.arange(6))),
                                  np.sin(n1 + np.radians(60 * np.arange(6)))])
    
    n2 = np.radians(a2)
    normals2 = np.column_stack([np.cos(n2 + np.radians(60 * np.arange(6))),
                                  np.sin(n2 + np.radians(60 * np.arange(6)))])
    
    all_normals = np.vstack([normals1, normals2])
    
    verts1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    verts2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        if proj1.max() + eps < proj2.min() or proj2.max() + eps < proj1.min():
            return True
    
    return False


def _validate(inner, outer, s, eps=1e-6):
    """Full validation of packing."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _separation_check(inner[i, :2], inner[i, 2], inner[j, :2], inner[j, 2], eps):
                return False
    
    # Check containment
    for i in range(12):
        cx, cy, aang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, aang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def _build_honeycomb(r_vec, a_vec, outer_rot_deg):
    """
    Build a honeycomb configuration from radial distances and angular deviations.
    r_vec: (12,) radial distances from center
    a_vec: (12,) angular deviations from ideal hexagonal lattice (in degrees)
    outer_rot_deg: rotation of outer hexagon
    Returns (inner, s) or None if invalid.
    """
    sqrt3 = np.sqrt(3)
    
    # Ideal hexagonal lattice positions for 12 hexagons
    # Center at (0,0), 6 neighbors at distance 2, 5 more at distance ~3.46
    # Axial coords: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
    #                (2,0), (2,-1), (1,-2), (0,-2), (-1,-2)
    axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
             (2,0), (2,-1), (1,-2), (0,-2), (-1,-2)]
    
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, sqrt3])
    
    ideal_centers = np.array([p[0]*e1 + p[1]*e2 for p in axial])
    
    # Compute ideal polar coordinates
    ideal_r = np.linalg.norm(ideal_centers, axis=1)
    ideal_theta = np.arctan2(ideal_centers[:, 1], ideal_centers[:, 0])
    
    # For center (0,0), r=0, theta is undefined. Set theta=0 for it.
    mask_zero = ideal_r < 1e-9
    ideal_theta[mask_zero] = 0.0
    ideal_r[mask_zero] = 1e-9  # Avoid division by zero
    
    # Apply radial and angular deviations
    r_actual = ideal_r * r_vec
    theta_actual = ideal_theta + np.radians(a_vec)
    
    centers = np.column_stack([r_actual * np.cos(theta_actual), r_actual * np.sin(theta_actual)])
    angles = np.zeros(12)
    
    # For the center hexagon, keep it at origin
    if mask_zero.any():
        centers[mask_zero] = [0.0, 0.0]
    
    s, ox, oy = _min_outer_side(centers, angles, outer_rot_deg)
    
    inner = np.column_stack([centers, angles])
    outer = np.array([ox, oy, outer_rot_deg])
    
    return inner, outer, s


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Uses a topologically constrained honeycomb representation.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    sqrt3 = np.sqrt(3)
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_config(inner, outer, s):
        nonlocal best_s, best_inner, best_outer
        if s < best_s:
            if _validate(inner, outer, s):
                best_s = s
                best_inner = inner
                best_outer = outer
    
    # Initial honeycomb configuration
    # r_vec: radial scale factors (1.0 = ideal)
    # a_vec: angular deviations in degrees
    r0 = np.ones(12)
    a0 = np.zeros(12)
    
    # Coarse search over outer rotation
    for rot in np.linspace(0, 60, 13):
        if time.time() > deadline - 120:
            break
        inner, outer, s = _build_honeycomb(r0, a0, rot)
        try_config(inner, outer, s)
    
    # Optimize radial and angular parameters for each outer rotation
    for rot in [0, 30, 15, 45, 10, 20, 40, 50]:
        if time.time() > deadline - 60:
            break
        
        def obj_honeycomb(params):
            if time.time() > deadline - 50:
                return 1e10
            r_vec = params[:12]
            a_vec = params[12:24]
            
            # Clamp radial values to reasonable range
            r_vec = np.clip(r_vec, 0.8, 1.5)
            a_vec = np.clip(a_vec, -30, 30)
            
            inner, outer, s = _build_honeycomb(r_vec, a_vec, rot)
            
            # Check separation
            for i in range(12):
                for j in range(i+1, 12):
                    if not _separation_check(inner[i, :2], inner[i, 2], inner[j, :2], inner[j, 2], 1e-7):
                        return 1e10
            
            return s
        
        try:
            res = minimize(obj_honeycomb, np.concatenate([r0, a0]), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < 1e9:
                r_opt = res.x[:12]
                a_opt = res.x[12:24]
                inner, outer, s = _build_honeycomb(r_opt, a_opt, rot)
                try_config(inner, outer, s)
        except Exception:
            pass
    
    # Fine-tune: try slight expansions for strict separation
    if time.time() < deadline - 30:
        for scale in [1.0005, 1.001, 1.0015, 1.002]:
            if time.time() > deadline - 20:
                break
            r_scaled = np.ones(12) * scale
            for rot in np.linspace(0, 60, 7):
                if time.time() > deadline - 10:
                    break
                inner, outer, s = _build_honeycomb(r_scaled, np.zeros(12), rot)
                try_config(inner, outer, s)
    
    # Fallback: standard hexagonal lattice
    if best_inner is None:
        lattice_points = np.array([
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2),
        ])
        e1 = np.array([2.0, 0.0])
        e2 = np.array([1.0, sqrt3])
        centers = np.array([p[0]*e1 + p[1]*e2 for p in lattice_points])
        angles = np.zeros(12)
        s, ox, oy = _min_outer_side(centers, angles, 0)
        best_s = s
        best_inner = np.column_stack([centers, angles])
        best_outer = np.array([ox, oy, 0.0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
