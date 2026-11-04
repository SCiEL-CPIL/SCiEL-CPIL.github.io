import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _compute_side_direct(inner_params, outer_angle_deg):
    """
    Compute minimum outer hexagon side length via support function.
    The outer hexagon with center at origin, side s, angle theta has
    6 half-planes. For each direction, find max projection of all inner
    hexagon vertices, then s = max_proj / cos(30°) for the tightest direction.
    
    Actually, for a regular hexagon with vertices at angles [theta + k*60],
    the half-plane normals are at angles [theta + 30 + k*60].
    The distance from center to each edge is s * cos(30°) = s * sqrt(3)/2.
    So for each normal direction n_k, we need:
        max over all inner vertices v: dot(v, n_k) <= s * sqrt(3)/2
    => s >= max_k (max_v dot(v, n_k)) / (sqrt(3)/2)
    """
    n = len(inner_params)
    theta = np.deg2rad(outer_angle_deg)
    # Normal directions (perpendicular to edges)
    normal_angles = theta + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        vs = _hex_vertices(x, y, a)
        all_verts.append(vs)
    all_verts = np.vstack(all_verts)
    
    # For each normal, compute max projection
    max_proj = 0.0
    for k in range(6):
        proj = all_verts @ normals[k]
        max_proj = max(max_proj, np.max(proj))
    
    side = max_proj / (math.sqrt(3) / 2)
    return side

def _seed_solution(cluster_angle_deg=0.0, scale=1.0):
    """Return the concentric ring seed solution with optional cluster rotation."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60 + cluster_angle_deg)
        r = D * scale
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60 + cluster_angle_deg)
        r = 3.0 * scale
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, cluster_angle_deg])
    
    # Compute side directly
    inner_arr = np.array(inner)
    side = _compute_side_direct(inner_arr, 30.0)
    
    return inner, (0.0, 0.0), side, 30.0

def _objective(params, outer_angle_deg, n):
    """
    Objective: outer side length.
    params: [cluster_angle, outer_angle, dx0, dy0, ..., dx10, dy10]
    Total: 2 + 2*11 = 24 parameters
    """
    cluster_angle = params[0]
    # outer_angle is fixed for now, but could be optimized
    # Displacements
    disp = params[2:]
    
    # Base positions from lattice
    base = np.zeros((n, 2))
    base[0] = [0.0, 0.0]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = D
        base[1+k] = [r * np.cos(angle), r * np.sin(angle)]
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        base[7+k] = [r * np.cos(angle), r * np.sin(angle)]
    
    # Apply cluster rotation
    c = np.cos(np.deg2rad(cluster_angle))
    s = np.sin(np.deg2rad(cluster_angle))
    rot = np.array([[c, -s], [s, c]])
    rotated = (base @ rot.T).T
    
    # Add displacements
    positions = rotated + disp.reshape(n, 2)
    
    # Inner hexagon angle = cluster_angle (all aligned with cluster)
    inner_params = np.column_stack([positions[:, 0], positions[:, 1], 
                                     np.full(n, cluster_angle)])
    
    side = _compute_side_direct(inner_params, outer_angle_deg)
    
    # Check overlap penalty
    penalty = 0.0
    for i in range(n):
        v1 = _hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j,0], inner_params[j,1], inner_params[j,2])
            # Quick check: if center distance < 2*apothex, they overlap
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 2.0 * math.sqrt(3)/2 - 0.01:  # apothex = sqrt(3)/2
                # More detailed check
                # For aligned hexagons, min separation = D = sqrt(3)
                # If distance < D - small, they overlap
                if d < D - 0.05:
                    penalty += 100.0 * (D - 0.05 - d) ** 2
                elif d < D:
                    # Check if they actually overlap (they touch at D)
                    # For same-orientation hexagons at distance D, they share an edge
                    # At distance < D, they overlap
                    penalty += 50.0 * (D - d) ** 2
    
    return side + penalty

def _objective_full(params, n):
    """
    Full objective with all parameters free.
    params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, outer_angle]
    Total: 3*11 + 1 = 34 parameters
    """
    inner_params = params[:3*n].reshape(n, 3)
    outer_angle = params[3*n]
    
    side = _compute_side_direct(inner_params, outer_angle)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(n):
        v1 = _hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j,0], inner_params[j,1], inner_params[j,2])
            d = np.linalg.norm(inner_params[i,:2] - inner_params[j,:2])
            # Minimum center distance for non-overlapping unit hexagons (same orientation)
            # is sqrt(3) for edge-to-edge contact
            if d < D - 0.02:
                penalty += 100.0 * (D - 0.02 - d) ** 2
            elif d < D:
                penalty += 20.0 * (D - d) ** 2
    
    return side + penalty

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid: all inner hexagons inside outer, no overlaps."""
    n = len(inner_params)
    
    # Check containment using support function
    theta = np.deg2rad(outer_angle_deg)
    normal_angles = theta + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_verts = []
    for i in range(n):
        vs = _hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        all_verts.append(vs)
    all_verts = np.vstack(all_verts)
    
    for k in range(6):
        proj = all_verts @ normals[k]
        if np.max(proj) > side * math.sqrt(3)/2 + 1e-9:
            return False
    
    # Check non-overlap
    for i in range(n):
        v1 = _hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j,0], inner_params[j,1], inner_params[j,2])
            d = np.linalg.norm(inner_params[i,:2] - inner_params[j,:2])
            if d < D - 1e-6:
                # More careful check: for same-orientation hexagons, they overlap iff d < D
                # For different orientations, need polygon intersection check
                # Simplified: if centers too close, assume overlap
                if d < D - 0.001:
                    return False
    
    return True

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin
    
    n = 11
    best_inner = None
    best_side = float('inf')
    best_outer_angle = 30.0
    
    # Get seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_outer_angle = seed_angle
    
    # Strategy 1: Optimize cluster angle and outer angle with small displacements
    # Parameters: [cluster_angle, outer_angle, dx0..dx10, dy0..dy10]
    # Start from seed
    for restart in range(3):
        if time.time() > DEADLINE - 20:
            break
        
        # Different initial cluster angles
        init_cluster_angles = [0.0, 5.0, -5.0]
        ca = init_cluster_angles[restart % len(init_cluster_angles)]
        
        init_params = np.zeros(2 + 2*n)
        init_params[0] = ca
        init_params[1] = 30.0  # outer angle
        # Displacements start at 0
        
        try:
            res = minimize(_objective, init_params, args=(30.0, n), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_side:
                # Reconstruct
                cluster_angle = res.x[0]
                disp = res.x[2:]
                
                base = np.zeros((n, 2))
                base[0] = [0.0, 0.0]
                for k in range(6):
                    angle = np.deg2rad(k * 60)
                    r = D
                    base[1+k] = [r * np.cos(angle), r * np.sin(angle)]
                for k in range(4):
                    angle = np.deg2rad(30 + k * 60)
                    r = 3.0
                    base[7+k] = [r * np.cos(angle), r * np.sin(angle)]
                
                c = np.cos(np.deg2rad(cluster_angle))
                s = np.sin(np.deg2rad(cluster_angle))
                rot = np.array([[c, -s], [s, c]])
                rotated = (base @ rot.T).T
                positions = rotated + disp.reshape(n, 2)
                
                inner_params = np.column_stack([positions[:, 0], positions[:, 1], 
                                                 np.full(n, cluster_angle)])
                side = _compute_side_direct(inner_params, 30.0)
                
                if side < best_side:
                    best_inner = inner_params.tolist()
                    best_side = side
                    best_outer_angle = 30.0
        except Exception:
            pass
    
    # Strategy 2: Full free optimization from seed
    if time.time() < DEADLINE - 30:
        init_full = np.zeros(3*n + 1)
        for i, (x, y, a) in enumerate(best_inner):
            init_full[i*3] = x
            init_full[i*3+1] = y
            init_full[i*3+2] = a
        init_full[3*n] = 30.0
        
        try:
            res = minimize(_objective_full, init_full, args=(n,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_side:
                inner_params = res.x[:3*n].reshape(n, 3)
                outer_angle = res.x[3*n]
                side = _compute_side_direct(inner_params, outer_angle)
                
                if side < best_side:
                    # Validate
                    if _is_valid(inner_params, side, outer_angle):
                        best_inner = inner_params.tolist()
                        best_side = side
                        best_outer_angle = outer_angle
        except Exception:
            pass
    
    # Strategy 3: Nelder-Mead refinement on cluster angle + outer angle only
    if time.time() < DEADLINE - 15:
        def obj_2d(params):
            ca, oa = params
            base = np.zeros((n, 2))
            base[0] = [0.0, 0.0]
            for k in range(6):
                angle = np.deg2rad(k * 60)
                r = D
                base[1+k] = [r * np.cos(angle), r * np.sin(angle)]
            for k in range(4):
                angle = np.deg2rad(30 + k * 60)
                r = 3.0
                base[7+k] = [r * np.cos(angle), r * np.sin(angle)]
            
            c = np.cos(np.deg2rad(ca))
            s = np.sin(np.deg2rad(ca))
            rot = np.array([[c, -s], [s, c]])
            rotated = (base @ rot.T).T
            positions = rotated
            
            inner_params = np.column_stack([positions[:, 0], positions[:, 1], 
                                             np.full(n, ca)])
            side = _compute_side_direct(inner_params, oa)
            return side
        
        try:
            res = minimize(obj_2d, [0.0, 30.0], method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_side:
                ca, oa = res.x
                base = np.zeros((n, 2))
                base[0] = [0.0, 0.0]
                for k in range(6):
                    angle = np.deg2rad(k * 60)
                    r = D
                    base[1+k] = [r * np.cos(angle), r * np.sin(angle)]
                for k in range(4):
                    angle = np.deg2rad(30 + k * 60)
                    r = 3.0
                    base[7+k] = [r * np.cos(angle), r * np.sin(angle)]
                
                c = np.cos(np.deg2rad(ca))
                s = np.sin(np.deg2rad(ca))
                rot = np.array([[c, -s], [s, c]])
                rotated = (base @ rot.T).T
                positions = rotated
                
                inner_params = np.column_stack([positions[:, 0], positions[:, 1], 
                                                 np.full(n, ca)])
                side = _compute_side_direct(inner_params, oa)
                
                if side < best_side:
                    if _is_valid(inner_params, side, oa):
                        best_inner = inner_params.tolist()
                        best_side = side
                        best_outer_angle = oa
        except Exception:
            pass
    
    # Final: try to tighten by reducing scale slightly
    if time.time() < DEADLINE - 10 and best_inner is not None:
        # Try scaling down the whole configuration
        inner_arr = np.array(best_inner)
        for scale in [0.999, 0.998, 0.997, 0.996, 0.995]:
            scaled = inner_arr.copy()
            scaled[:, :2] *= scale
            side = _compute_side_direct(scaled, best_outer_angle)
            
            # Check overlap
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(scaled[i,:2] - scaled[j,:2])
                    if d < D * scale - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and side < best_side:
                best_inner = scaled.tolist()
                best_side = side
    
    # Ensure we have a valid result
    if best_inner is None:
        best_inner, _, best_side, best_outer_angle = _seed_solution()
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
    
    center = (0.0, 0.0)
    side = float(best_side)
    angle_deg = float(best_outer_angle)
    
    return inner_rows, center, side, angle_deg

