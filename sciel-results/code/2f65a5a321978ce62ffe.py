"""Combined program: LP-based enclosing hexagon with configuration search."""
import math
import time
import numpy as np
import itertools
from scipy.optimize import linprog, minimize

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
    
    # b_ub for each direction k: -all_vertices @ normals[k]
    rhs = -(all_vertices @ normals.T)  # (n_pts, 6)
    b_ub_parts = rhs.T  # (6, n_pts)
    b_ub = b_ub_parts.flatten()  # (6 * n_pts,)
    
    # A_ub: repeat each normal row n_pts times
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    # Add r >= 0 constraint
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        # Fallback: use centroid and compute required r
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T  # (n_pts, 6)
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute all vertices once
    all_vertices = _compute_all_vertices(inner_hexes)
    
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
    
    # Fourth pass: very fine sweep
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        angles_very_fine = np.arange(max(0.0, best_angle - 0.5), min(60.0, best_angle + 0.5), 0.01)
        for angle_deg in angles_very_fine:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Fifth pass: Nelder-Mead polish on angle only
    if best_side < float('inf') and time.time() < _DEADLINE - 10.0:
        from scipy.optimize import minimize_scalar
        
        def objective_angle(ang_p):
            ang_p = ang_p % 60.0
            _, _, side_p, _ = _min_enclosing_hex_side_lp(inner_hexes, float(ang_p), all_vertices)
            return side_p
        
        try:
            bounds = (max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0))
            res = minimize_scalar(objective_angle, bounds=bounds, method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 100})
            if res.success:
                ang_opt = float(res.x % 60.0)
                cx_v, cy_v, side_v, _ = _min_enclosing_hex_side_lp(inner_hexes, ang_opt, all_vertices)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_opt
        except Exception:
            pass
    
    # Sixth pass: final high-precision polish
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        for delta in [0.001, -0.001, 0.002, -0.002, 0.005, -0.005]:
            ang_test = (best_angle + delta) % 60.0
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(ang_test), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(ang_test)
    
    return best_center[0], best_center[1], best_side, best_angle

def _evaluate_configuration(positions, inner_angles, all_vertices=None):
    """Evaluate a configuration of inner hexagons and return the enclosing hexagon side length."""
    if all_vertices is None:
        inner_hexes = [[pos[0], pos[1], ang] for pos, ang in zip(positions, inner_angles)]
        all_vertices = _compute_all_vertices(inner_hexes)
    
    # Quick coarse sweep to estimate
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    angles_coarse = np.arange(0, 60, 5.0)
    for angle_deg in angles_coarse:
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    return best_side, best_center, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        sqrt3 = math.sqrt(3)
        scale = 1.001
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))
        
        candidates.sort(key=lambda t: t[0])
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        
        inner = [[pos[0], pos[1], 30.0] for pos in positions]
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    sqrt3 = math.sqrt(3)
    
    # Scale factor to ensure strict disjointness
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    # Base positions: first 7 (center + ring 1)
    base_positions = [(candidates[i][1], candidates[i][2]) for i in range(7)]
    
    # Ring 2 positions: next candidates (distance 2 from center)
    ring2_positions = [(candidates[i][1], candidates[i][2]) for i in range(7, 21)]
    
    # We need to choose 4 more from ring 2 to make 11 total
    # Try different combinations and evaluate with LP
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_angles = [30.0] * N
    
    # Phase 1: Try different combinations of 4 from ring 2
    ring2_indices = list(range(len(ring2_positions)))
    all_combos = list(itertools.combinations(ring2_indices, 4))
    
    # Limit the number of combinations to try based on time
    max_combos = min(len(all_combos), 50)
    
    for combo in all_combos[:max_combos]:
        if time.time() > _DEADLINE - 30.0:
            break
        
        selected_r2 = [ring2_positions[i] for i in combo]
        full_positions = base_positions + selected_r2
        
        # Create inner hexagons with angle 30 (pointy-top)
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in full_positions]
        
        # Quick evaluation with coarse angle sweep
        all_vertices = _compute_all_vertices(inner_hexes)
        
        best_side_combo = float('inf')
        best_center_combo = (0.0, 0.0)
        best_angle_combo = 0.0
        
        angles_coarse = np.arange(0, 60, 5.0)
        for angle_deg in angles_coarse:
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side_combo:
                best_side_combo = side
                best_center_combo = (cx, cy)
                best_angle_combo = float(angle_deg)
        
        if best_side_combo < best_side:
            best_side = best_side_combo
            best_positions = full_positions
            best_center = best_center_combo
            best_angle = best_angle_combo
    
    # Phase 2: Refine the best configuration with full angle optimization
    if best_positions is not None:
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in best_positions]
        
        # Use the full multi-scale angle optimization
        cx, cy, side, angle = _min_enclosing_hex_side(inner_hexes)
        
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = angle
    
    # Phase 3: Try alternative inner hexagon orientations
    # Sometimes rotating inner hexagons can help with the enclosing hexagon
    if time.time() < _DEADLINE - 20.0:
        # Try angle 0 (flat-top) for inner hexagons
        inner_hexes_flat = [[pos[0], pos[1], 0.0] for pos in best_positions]
        cx, cy, side, angle = _min_enclosing_hex_side(inner_hexes_flat)
        
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = angle
            best_inner_angles = [0.0] * N
    
    # Phase 4: Local optimization of inner positions
    # Try small perturbations of the inner hexagon positions
    if best_positions is not None and time.time() < _DEADLINE - 15.0:
        # Create a parameter vector for optimization
        # Each inner hexagon has (x, y) position, angle fixed at 30
        
        def objective(params):
            positions_opt = []
            for i in range(N):
                x = params[i * 2]
                y = params[i * 2 + 1]
                positions_opt.append((x, y))
            
            inner_hexes_opt = [[pos[0], pos[1], 30.0] for pos in positions_opt]
            all_vertices_opt = _compute_all_vertices(inner_hexes_opt)
            
            # Quick LP evaluation at a few angles
            best_s = float('inf')
            for ang in [0.0, 15.0, 30.0, 45.0]:
                _, _, s, _ = _min_enclosing_hex_side_lp(inner_hexes_opt, ang, all_vertices_opt)
                if s < best_s:
                    best_s = s
            
            return best_s
        
        # Initial parameters
        x0 = []
        for pos in best_positions:
            x0.append(pos[0])
            x0.append(pos[1])
        
        # Use SLSQP with bounds to keep hexagons disjoint
        # For simplicity, use small bounds around initial positions
        bounds = []
        for i in range(N):
            bounds.append((best_positions[i][0] - 0.1, best_positions[i][0] + 0.1))
            bounds.append((best_positions[i][1] - 0.1, best_positions[i][1] + 0.1))
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-8})
            if res.success:
                positions_opt = []
                for i in range(N):
                    positions_opt.append((res.x[i * 2], res.x[i * 2 + 1]))
                
                inner_hexes_opt = [[pos[0], pos[1], 30.0] for pos in positions_opt]
                cx, cy, side, angle = _min_enclosing_hex_side(inner_hexes_opt)
                
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle
                    best_positions = positions_opt
        except Exception:
            pass
    
    # Create the final inner hexagons
    if best_positions is None:
        # Fallback to the original configuration
        positions = []
        for _, x, y in candidates:
            positions.append((x, y))
            if len(positions) == N:
                break
        best_positions = positions
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in best_positions]
        cx, cy, best_side, best_angle = _min_enclosing_hex_side(inner_hexes)
        best_center = (cx, cy)
    else:
        inner_hexes = [[pos[0], pos[1], 30.0] for pos in best_positions]
    
    # Final verification and return
    return inner_hexes, (best_center[0], best_center[1]), best_side, best_angle
