import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3.0)

    # Precompute outer hexagon normals (flat-top, centered at origin)
    outer_normals = np.array([[math.cos(math.radians(30 + 60*k)),
                               math.sin(math.radians(30 + 60*k))] for k in range(6)])

    def get_hex_verts_np(cx, cy, angle_deg):
        """6x2 vertex array for unit hexagon."""
        angles = np.radians(angle_deg + 60.0 * np.arange(6))
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def point_in_convex(pt, poly):
        """pt: 2d array, poly: 6x2 array. Convex polygon containment."""
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            edge = p2 - p1
            to_pt = pt - p1
            cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
            if cross < -1e-12:
                return False
        return True

    def pt_to_convex_boundary(pt, poly):
        """Distance from point to boundary of convex hexagon."""
        min_d = float('inf')
        for k in range(6):
            p1 = poly[k]
            p2 = poly[(k + 1) % 6]
            d = p2 - p1
            len_sq = float(d @ d)
            if len_sq < 1e-20:
                dist = float(np.linalg.norm(pt - p1))
            else:
                t = float(np.dot(pt - p1, d)) / len_sq
                t = max(0.0, min(1.0, t))
                proj = p1 + t * d
                dist = float(np.linalg.norm(pt - proj))
            if dist < min_d:
                min_d = dist
        return min_d

    def compute_L_and_penalty(params):
        """Compute (L, penalty) for given parameters."""
        verts_list = []
        all_pts = []
        for i in range(N):
            x, y, th = params[3*i], params[3*i+1], params[3*i+2]
            v = get_hex_verts_np(x, y, th)
            verts_list.append(v)
            all_pts.append(v)

        all_pts = np.vstack(all_pts)
        projections = all_pts @ outer_normals.T
        L = float(projections.max()) / (sqrt3 / 2.0)

        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                for k in range(6):
                    if point_in_convex(v1[k], v2):
                        penalty += pt_to_convex_boundary(v1[k], v2) ** 2
                    if point_in_convex(v2[k], v1):
                        penalty += pt_to_convex_boundary(v2[k], v1) ** 2
        return L, penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        L, pen = compute_L_and_penalty(params)
        # High penalty weight to strictly enforce non-overlap
        return L + 1000.0 * pen

    def separation_objective(params):
        """Objective to minimize overlap only, used for final repair."""
        _, pen = compute_L_and_penalty(params)
        return pen

    # Initializations
    grid_centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                grid_centers.append((x, y))

    dists = [(math.hypot(x, y), i) for i, (x, y) in enumerate(grid_centers)]
    dists.sort()
    init_grid_idx = [d[1] for d in dists[:12]]

    params_grid = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        params_grid.extend([x, y, 0.0])

    # Ring-based: 1 center + 5 inner + 6 outer? 
    # Let's try a more symmetric start: 1 center, 6 around it, 5 more?
    # Or just stick to the grid which is dense.
    
    # Random
    np.random.seed(42)
    params_random = []
    for _ in range(N):
        params_random.extend([np.random.uniform(-2.5, 2.5),
                              np.random.uniform(-2.5, 2.5),
                              np.random.uniform(0, 360)])

    best_params = None
    best_L = float('inf')
    best_pen = float('inf')

    def try_optimize(init_params, maxiter, xatol, fatol):
        nonlocal best_params, best_L, best_pen
        if time.time() > deadline - 15:
            return
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': xatol,
                                    'fatol': fatol, 'adaptive': True})
            if time.time() > deadline - 10:
                return
            L, pen = compute_L_and_penalty(res.x)
            # Accept if penalty is very low (disjoint)
            if pen < 1e-5 and L < best_L:
                best_L = L
                best_pen = pen
                best_params = res.x.copy()
            # Also keep track of best penalty if we haven't found a disjoint one yet
            elif pen < best_pen:
                best_pen = pen
                if best_params is None:
                    best_params = res.x.copy()
        except Exception:
            pass

    # Multi-start with diverse initializations
    try_optimize(params_grid, 5000, 1e-6, 1e-6)
    try_optimize(params_random, 5000, 1e-6, 1e-6)

    # If we found a candidate, try to polish it
    if best_params is not None and time.time() < deadline - 30:
        # Polish L
        try_optimize(best_params, 3000, 1e-8, 1e-8)
        
        # Check if it's disjoint
        _, final_pen = compute_L_and_penalty(best_params)
        
        # If there is still overlap, run a separation pass
        if final_pen > 1e-5:
            # Minimize penalty only to separate hexagons
            try:
                sep_res = minimize(separation_objective, best_params, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-8,
                                            'fatol': 1e-8, 'adaptive': True})
                # If separation was successful, update params
                if sep_res.fun < 1e-5:
                    best_params = sep_res.x
                    best_L, _ = compute_L_and_penalty(best_params)
            except Exception:
                pass

    # Fallback: If we still have overlaps or no best_params, use a simple valid grid
    # A grid of 12 hexagons can fit in a hexagon of side 4.
    # Centers at (0,0), (1.5, sqrt3/2), etc.
    if best_params is None or compute_L_and_penalty(best_params)[1] > 1e-5:
        # Construct a simple valid configuration
        # 3x4 grid of hexagons? 
        # Side length 1 hexagons.
        # Distance between centers in grid: 1.5 in x, sqrt(3) in y (for offset rows)
        # Let's place them in a 2x6 or 3x4 block.
        # 3 rows, 4 cols.
        # Row 0: y=0, x = -1.5, 0, 1.5, 3.0
        # Row 1: y=sqrt(3), x = -0.75, 0.75, 2.25, 3.75
        # Row 2: y=2*sqrt(3), x = -1.5, 0, 1.5, 3.0
        # This is a parallelogram. Bounding hexagon needs to be larger.
        # Let's just use the initial grid params which are likely valid if L is large enough.
        # But we need to minimize L.
        # If optimization failed, return the grid initial which is likely valid for L~4.
        best_params = np.array(params_grid, dtype=float)
        best_L, _ = compute_L_and_penalty(best_params)

    positions = []
    for i in range(N):
        positions.append([float(best_params[3*i]),
                          float(best_params[3*i+1]),
                          float(best_params[3*i+2])])

    return positions, (0.0, 0.0), best_L, 0.0
