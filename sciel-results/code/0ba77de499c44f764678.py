import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
EPS = 1e-9
MIN_DIST = math.sqrt(3.0) # Distance between centers of touching unit hexagons

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate candidate configurations (axial coordinates for aligned hexagons)
    # These serve as initial seeds for the continuous optimizer
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    
    # Convert candidates to Cartesian points
    init_configs = []
    for axial in candidates:
        pts = axial_to_cartesian(axial)
        # Center the configuration
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        init_configs.append(pts_centered)

    def objective(params):
        """
        params: [x1, y1, ..., x11, y11, theta]
        Returns: Side length S + penalty for overlaps
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        positions = params[:2*N].reshape(N, 2)
        theta = params[2*N]

        # Normals for the 6 sides of the outer hexagon
        # The outer hexagon is rotated by theta.
        # The normals are perpendicular to the sides.
        # For a regular hexagon with side S, the apothem is S * sqrt(3)/2.
        # The projection of the inner hexagons onto the normal must be <= apothem.
        # We want to minimize S such that max_projection <= S * sqrt(3)/2.
        # S = max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3).
        
        # To handle the centering, we can shift positions so their centroid is at 0?
        # No, the outer hexagon center is free. 
        # Actually, for a given set of points, the minimal enclosing hexagon (for a fixed orientation) 
        # is centered at the point that minimizes the max projection.
        # However, since we are optimizing positions AND rotation, we can just compute the width.
        # If we assume the outer hexagon is centered at the origin, the constraint is:
        # max_i (proj_i) - min_i (proj_i) + 2*apothem_inner <= 2*apothem_outer
        # Width + sqrt(3) <= S * sqrt(3)
        # S >= (Width + sqrt(3)) / sqrt(3) = Width/sqrt(3) + 1
        
        # Let's calculate widths in 6 directions.
        angles = theta + np.pi/6.0 + np.arange(6) * np.pi/3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Projections of centers
        projs = positions @ normals.T # Shape (6, N)
        
        # Width in each direction
        widths = np.max(projs, axis=1) - np.min(projs, axis=1)
        
        # The required side length S for this configuration and orientation
        # S_k = (width_k + sqrt(3)) / sqrt(3)
        # We take the max over k
        S_candidate = np.max(widths) / math.sqrt(3.0) + 1.0
        
        # Penalty for overlaps
        # Distance between all pairs
        # pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        # dists = np.linalg.norm(pos_diff, axis=2)
        # penalty = np.sum(np.maximum(0, MIN_DIST - dists)**2)
        
        # Efficient penalty calculation
        # Using broadcasting
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Only consider i < j to avoid double counting and self distance
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[i_idx, j_idx]
        pair_dists = np.sqrt(pair_dists_sq)
        violations = np.maximum(0, MIN_DIST - pair_dists)
        penalty = np.sum(violations**2) * 10.0 # Weight penalty
        
        return S_candidate + penalty

    def objective_exact(params):
        """
        Same as objective but without penalty, used for final polish if we trust the disjointness.
        """
        positions = params[:2*N].reshape(N, 2)
        theta = params[2*N]
        angles = theta + np.pi/6.0 + np.arange(6) * np.pi/3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = positions @ normals.T
        widths = np.max(projs, axis=1) - np.min(projs, axis=1)
        return np.max(widths) / math.sqrt(3.0) + 1.0

    best_S = float('inf')
    best_params = None

    # Run continuous optimization from each initial configuration
    for idx, init_pts in enumerate(init_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Flatten positions and add initial theta=0
        init_params = np.concatenate([init_pts.ravel(), [0.0]])
        
        # Try a few different initial rotations for the outer hexagon
        for init_theta in [0.0, math.pi/6.0, math.pi/12.0]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            init_params[2*N] = init_theta
            
            try:
                # Use Powell method for robustness with non-smooth penalty
                res = minimize(objective, init_params, method='Powell', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                
                if res.fun < best_S:
                    # Verify disjointness for the best result
                    pos = res.x[:2*N].reshape(N, 2)
                    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                    dists_sq = np.sum(diff**2, axis=2)
                    i_idx, j_idx = np.triu_indices(N, k=1)
                    pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
                    if np.min(pair_dists) >= MIN_DIST - 1e-6:
                        best_S = res.fun
                        best_params = res.x
            except Exception:
                pass

    if best_params is None:
        # Fallback to the best candidate evaluated without optimization
        # Or just return a simple valid configuration
        best_params = np.concatenate([init_configs[0].ravel(), [0.0]])
        best_S = objective(best_params)

    # Final Polish: Refine the best solution found
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            # Use Nelder-Mead for a final precise search
            res_polish = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_polish.fun < best_S:
                pos = res_polish.x[:2*N].reshape(N, 2)
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                i_idx, j_idx = np.triu_indices(N, k=1)
                pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
                if np.min(pair_dists) >= MIN_DIST - 1e-6:
                    best_S = res_polish.fun
                    best_params = res_polish.x
        except Exception:
            pass

    # Extract results
    positions = best_params[:2*N].reshape(N, 2)
    theta = best_params[2*N]
    
    # Calculate the exact minimal enclosing hexagon for these positions
    # The formula S = max_width/sqrt(3) + 1 assumes the hexagon is centered at the midpoint of the min/max projections.
    # This is correct for the minimal enclosing hexagon for a fixed orientation.
    
    # However, we should center the whole configuration at the origin for the output
    # The problem asks for (inner, center, side, angle).
    # The 'center' is the center of the outer hexagon.
    # In our optimization, we implicitly assumed the outer hexagon is centered at the origin?
    # No, the width calculation is translation invariant.
    # The "center" of the minimal enclosing hexagon for a fixed orientation is the average of the min and max projections in each direction?
    # Actually, for a regular hexagon, the center is unique for a given orientation if the shape is not degenerate.
    # Let's find the center (cx, cy) that minimizes the max projection.
    # This is a linear programming problem, but we can approximate it.
    # For the 6 normals, the center (cx, cy) should satisfy:
    # cx*nx + cy*ny + D >= proj_i for all i
    # cx*nx + cy*ny - D <= proj_i for all i
    # Minimize D.
    # Since we already have the width, D = max_width/2.
    # The center is the point such that the projections are balanced.
    # A good approximation for the center is the centroid of the points, adjusted.
    # Let's just use the centroid of the inner hexagons as the outer center.
    # This is valid if the packing is roughly symmetric.
    # To be safer, we can compute the exact center using the LP from the original code if time permits.
    
    cx, cy = np.mean(positions, axis=0)
    
    # Recalculate S and center precisely
    # Let's use the LP method for the final answer to ensure validity
    def calc_S_and_center_for_theta(pts, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        from scipy.optimize import linprog
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx_res = res.x[0]
                cy_res = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx_res, cy_res)
            else:
                return float('inf'), (0,0)
        except Exception:
            return float('inf'), (0,0)

    S_final, (cx_final, cy_final) = calc_S_and_center_for_theta(positions, theta)
    
    if S_final == float('inf'):
        # Fallback
        S_final = best_S
        cx_final, cy_final = cx, cy

    # Ensure disjointness with a tiny scale up if necessary (should not be needed if penalty worked)
    # But let's check
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
    if np.min(pair_dists) < MIN_DIST - 1e-8:
        # Scale up positions slightly around centroid
        centroid = np.mean(positions, axis=0)
        scale = (MIN_DIST - 1e-9) / np.min(pair_dists)
        positions = centroid + (positions - centroid) * scale
        # Recalculate S
        S_final, (cx_final, cy_final) = calc_S_and_center_for_theta(positions, theta)

    inner = []
    for x, y in positions:
        inner.append([float(x), float(y), 0.0]) # Inner hexagons aligned (angle 0)
        
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(math.degrees(theta))

    return inner, center, side, rotation

