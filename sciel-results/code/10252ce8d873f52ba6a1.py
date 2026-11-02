"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
import itertools

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    Vectorized constraint construction for speed.
    """
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # For each of 6 directions, compute nx, ny
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Vectorized: for each direction k, constraint is:
    #   -nx_k * cx - ny_k * cy - r <= -vx*nx_k - vy*ny_k for all vertices
    # A_ub has shape (6 * n_pts + 1, 3), b_ub has shape (6 * n_pts + 1,)
    
    # For each direction, the RHS for all vertices is: -(all_vertices @ [nx, ny])
    # A_ub rows: for direction k, vertex i: [-nx_k, -ny_k, -1]
    # We can build this efficiently
    
    # b_ub for each direction k: -all_vertices @ normals[k]
    # Shape: (6, n_pts) -> transpose to (n_pts, 6) -> reshape
    rhs = -(all_vertices @ normals.T)  # (n_pts, 6)
    b_ub_parts = rhs.T  # (6, n_pts)
    b_ub = b_ub_parts.flatten()  # (6 * n_pts,)
    
    # A_ub: repeat each normal row n_pts times
    # normals is (6, 2), we need (6*n_pts, 3)
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    # Add r >= 0 constraint
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        # Fallback: use centroid and compute required r
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        # r = max over all vertices and directions of dot product
        # (v - c) @ n for all v, all n
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T  # (n_pts, 6)
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _quick_min_enclosing_hex_side(inner_hexes):
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Coarse sweep
    angles_coarse = np.arange(0, 60, 2.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
            
    # Fine sweep
    if best_side < float('inf'):
        angles_fine = np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
                
    # Finer sweep
    if best_side < float('inf'):
        angles_finer = np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
                
    return best_center[0], best_center[1], best_side, best_angle

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute all vertices once
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Try a range of angles for the outer hexagon
    # Due to 6-fold symmetry, we only need to search 0 to 60 degrees.
    
    # First pass: coarse sweep
    angles_coarse = np.arange(0, 60, 2.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Second pass: fine sweep around the best angle
    if best_side < float('inf'):
        angles_fine = np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25)
        for angle_deg in angles_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        angles_finer = np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05)
        for angle_deg in angles_finer:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Fourth pass: Nelder-Mead polish on (cx, cy, angle) jointly
    # The LP gives optimal (cx, cy) for fixed angle, but we can refine
    # by doing a local search that re-solves the LP at each step.
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        from scipy.optimize import minimize
        
        def objective(params):
            cx_p, cy_p, ang_p = params
            # Normalize angle to [0, 60)
            ang_p = ang_p % 60.0
            # Solve LP for this angle
            _, _, side_p, _ = _min_enclosing_hex_side_lp(inner_hexes, float(ang_p), all_vertices)
            return side_p
        
        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.success:
                cx_nm, cy_nm, ang_nm = res.x
                ang_nm = float(ang_nm % 60.0)
                # Verify with LP
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(inner_hexes, ang_nm, all_vertices)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_nm
        except Exception:
            pass
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    # Lattice points are (i, j) -> i*u + j*v
    # Let's generate points with |i|, |j| <= 3 and sort by distance
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Extract the 7 central points and the next 6
    # The first 7 are the "flower" (1 center + 6 neighbors)
    central_pts = [ (x, y) for _, x, y in candidates[:7] ]
    # The next 6 are the candidates for the 2nd ring
    second_ring_pts = [ (x, y) for _, x, y in candidates[7:13] ]
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # Try all combinations of 4 from the 6 second ring points
    # 15 combinations
    for combo in itertools.combinations(second_ring_pts, 4):
        pts = central_pts + list(combo)
        inner_candidate = [[p[0], p[1], 30.0] for p in pts]
        
        # Compute min enclosing hex side for this cluster
        cx, cy, side, angle = _quick_min_enclosing_hex_side(inner_candidate)
        
        if side < best_side:
            best_side = side
            best_inner = inner_candidate
            best_outer = (cx, cy, angle)
            
    # Now polish the best one
    if best_inner is not None:
        cx, cy, side, angle = _min_enclosing_hex_side(best_inner)
        # Update best if polish improved it
        if side < best_side:
            best_side = side
            best_outer = (cx, cy, angle)
    
    inner = best_inner
    
    # Fallback if something went wrong (should not happen)
    if inner is None:
        positions = [ (x, y) for _, x, y in candidates[:N] ]
        inner = [[pos[0], pos[1], 30.0] for pos in positions]
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
    else:
        cx, cy, angle = best_outer
        side = best_side
        
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
