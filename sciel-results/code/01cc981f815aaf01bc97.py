import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Leave some buffer
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)

    A_rows = []
    b_vals = []

    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    # R >= 0
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])

    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _objective_for_angle(verts, outer_angle_deg):
    """Return the side length for a given outer angle, or a large penalty."""
    res = _min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return 1e6
    return res[2]


def _build_positions(layout_idx, scale=1.01):
    """Generate 11 positions for given layout index with a safety scale."""
    sqrt3 = _SQRT3
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    # Layout 0: Center + Ring 1 (6) + 4 from Ring 2 (at cardinal directions relative to ring 1 gaps)
    # Ring 1 indices in axial coords (q, r) where x = q*u + r*v
    # (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # To be symmetric, pick 4 that form a square or cross.
    # Let's pick (2,0), (0,2), (-2,0), (0,-2) -> These are "opposite" in the hex lattice sense?
    # (2,0) is East. (0,2) is North-West? No.
    # Let's just pick the 4 closest to the "axes" of the bounding box.
    
    if layout_idx == 0:
        # Symmetric cross
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        # Pick 4 from ring 2 that are symmetric
        # (2,0) and (-2,0) are opposite. (0,2) and (0,-2) are opposite?
        # (0,2) is q=0, r=2. x = 0, y = 3. 
        # (2,0) is q=2, r=0. x = 2*sqrt3, y = 0.
        # This forms a diamond/cross shape.
        extra = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        target.extend(extra)
        
    elif layout_idx == 1:
        # Symmetric square-ish
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        extra = [(1, 1), (-1, 1), (-1, -1), (1, -1)] # Wait, these are ring 1 neighbors?
        # Ring 1 is distance 1. Ring 2 is distance 2.
        # (1,1) is dist 1? No, dist is max(|q|,|r|,|q+r|) ? No, hex distance is (|q|+|r|+|q+r|)/2.
        # (1,1): (1+1+2)/2 = 2. So (1,1) is Ring 2.
        # (1,-1): (1+1+0)/2 = 1. So (1,-1) is Ring 1.
        # Let's use axial coordinates properly.
        # Neighbors of (0,0): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1).
        # Ring 2:
        # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0) ... and the other half.
        # Let's pick (2,0), (0,2), (-2,0), (0,-2) for layout 0.
        # For layout 1, pick (2,-1), (-2,1), (-1,-2), (1,2)? 
        # (2,-1) dist: (2+1+1)/2 = 2.
        # Let's try a different set: (2,0), (1,1), (-1,2), (-2,1), (-1,0)... no.
        # Let's just use the 4 "corner" ones of the ring 2 hexagon.
        extra = [(2, -1), (-2, 1), (-1, -2), (1, 2)]
        target.extend(extra)
        
    elif layout_idx == 2:
        # 3-4-4 rows approximation
        # This is harder to do with axial coords directly.
        # Let's stick to axial for symmetry.
        # Maybe pick (2,0), (1,1), (0,2) and their opposites? That's 6.
        # We need 4.
        # Let's try (2,0), (-2,0), (1,1), (-1,-1) ? 
        # (1,1) is opposite to (-1,-1).
        extra = [(2, 0), (-2, 0), (1, 1), (-1, -1)]
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        target.extend(extra)

    else:
        # Fallback: just center + ring 1 + first 4 of ring 2
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        target.extend(ring2[:4])

    positions = []
    for q, r in target:
        x = q * u[0] + r * v[0]
        y = q * u[1] + r * v[1]
        positions.append((x, y))
    
    return positions[:N]


def _compute_cluster_vertices(positions, global_angle_deg):
    """Compute all vertices for a cluster of hexagons rotated by global_angle_deg."""
    all_vertices = []
    for px, py in positions:
        verts = _hex_vertices(px, py, global_angle_deg)
        all_vertices.append(verts)
    return np.vstack(all_vertices)


def _optimize_outer_angle(verts, initial_angle, time_limit):
    """
    Optimize the outer hexagon angle using exact LP objective.
    Returns (best_angle, best_side).
    """
    best_side = 1e10
    best_angle = initial_angle

    # Coarse grid first to find a good basin
    coarse_best = 1e10
    coarse_best_angle = initial_angle
    # Sample 0 to 30 degrees with step 1
    for a in np.arange(0, 30.5, 1.0):
        if time.time() > time_limit:
            break
        s = _objective_for_angle(verts, a)
        if s < coarse_best:
            coarse_best = s
            coarse_best_angle = a
            
    if coarse_best < best_side:
        best_side = coarse_best
        best_angle = coarse_best_angle

    # Local refinement
    lo = max(0.0, best_angle - 2.0)
    hi = min(30.0, best_angle + 2.0)
    
    # Use golden section or similar via minimize_scalar
    try:
        res_scalar = minimize_scalar(
            lambda a: _objective_for_angle(verts, a),
            bounds=(lo, hi),
            method='bounded',
            options={'xatol': 1e-8, 'maxiter': 100}
        )
        if res_scalar.success:
            if res_scalar.fun < best_side:
                best_side = res_scalar.fun
                best_angle = float(res_scalar.x)
    except Exception:
        pass
        
    # Final fine polish
    if time.time() < time_limit:
        for delta_max, step in [(0.01, 0.001), (0.001, 0.0001)]:
            if time.time() > time_limit:
                break
            lo2 = max(0.0, best_angle - delta_max)
            hi2 = min(30.0, best_angle + delta_max)
            n_steps = int((hi2 - lo2) / step) + 1
            if n_steps > 500:
                step = (hi2 - lo2) / 500
            for a in np.arange(lo2, hi2 + step / 2, step):
                if time.time() > time_limit:
                    break
                s = _objective_for_angle(verts, a)
                if s < best_side:
                    best_side = s
                    best_angle = float(a)

    return best_angle, best_side


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_global_angle = 0.0

    num_layouts = 3
    layouts = [_build_positions(i) for i in range(num_layouts)]

    # We will optimize the global rotation of the inner cluster
    # and the outer hexagon angle simultaneously (nested loop)
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break

        # Try a few global rotations for the inner cluster
        # The symmetry of the cluster is 60 degrees (if the extra 4 are symmetric).
        # Layout 0 (Cross) has 90 degree symmetry? No, hex lattice is 60.
        # The cluster of 11 (1+6+4 cross) has 4-fold symmetry in terms of "arms", 
        # but the underlying lattice is 6-fold.
        # Actually, (2,0), (0,2), (-2,0), (0,-2) in axial coords:
        # (2,0) -> (0,2) is a rotation?
        # Let's just sample 0 to 30 degrees for global rotation.
        
        # To save time, we can sample global rotations coarsely first.
        global_angles = np.arange(0, 30.0, 5.0)
        
        local_best_side = 1e10
        local_best_global_angle = 0.0
        local_best_outer_angle = 0.0
        local_best_center = (0,0)
        
        for g_ang in global_angles:
            if time.time() > _DEADLINE - 10.0:
                break
            
            verts = _compute_cluster_vertices(positions, g_ang)
            
            # Find best outer angle for this global rotation
            # Start outer angle search at 0
            outer_ang, side = _optimize_outer_angle(verts, 0.0, _DEADLINE - 12.0)
            
            if side < local_best_side:
                local_best_side = side
                local_best_global_angle = g_ang
                local_best_outer_angle = outer_ang
                res_final = _min_enclosing_hex_side(verts, outer_ang)
                if res_final:
                    local_best_center = (res_final[0], res_final[1])

        # If we found a good local best, refine the global angle
        if local_best_side < best_side:
            # Refine global angle around local_best_global_angle
            # We can do a scalar minimization on global angle
            # Objective: min over outer_angle of side_length(global_angle)
            # This is expensive. Let's just do a fine grid on global angle.
            
            refine_lo = max(0.0, local_best_global_angle - 5.0)
            refine_hi = min(30.0, local_best_global_angle + 5.0)
            
            for g_ang in np.arange(refine_lo, refine_hi + 0.5, 0.5):
                if time.time() > _DEADLINE - 8.0:
                    break
                verts = _compute_cluster_vertices(positions, g_ang)
                outer_ang, side = _optimize_outer_angle(verts, local_best_outer_angle, _DEADLINE - 10.0)
                
                if side < local_best_side:
                    local_best_side = side
                    local_best_global_angle = g_ang
                    local_best_outer_angle = outer_ang
                    res_final = _min_enclosing_hex_side(verts, outer_ang)
                    if res_final:
                        local_best_center = (res_final[0], res_final[1])

            if local_best_side < best_side:
                best_side = local_best_side
                best_global_angle = local_best_global_angle
                best_outer_angle = local_best_outer_angle
                best_center = local_best_center
                best_inner = [[pos[0], pos[1], best_global_angle] for pos in positions]

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
