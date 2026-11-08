import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 115.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    # Base angles for the 13 points
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Fourier modes: K=4 to allow more shape flexibility
    # Parameters: r0, a1, b1, a2, b2, a3, b3, a4, b4
    # Total 9 parameters.
    # We fix center at (0,0) to reduce dimensions, as the objective is translation invariant
    # after normalization, but we can include cx, cy for robustness if needed.
    # However, radial deformation around origin is sufficient if we normalize.
    # Let's include cx, cy to handle potential asymmetries not captured by radial r(theta)
    # if the "center" of the polygon shifts.
    # Actually, r(theta) with cx,cy allows ellipses etc.
    
    K = 4
    n_params = 2 + 1 + 2 * K  # cx, cy, r0, a1..bK
    
    def params_to_points(params):
        cx, cy, r0 = params[0], params[1], params[2]
        a = params[3:3+K]
        b = params[3+K:3+2*K]
        
        # Compute r(theta)
        r = np.full(N, r0)
        for k in range(1, K+1):
            r += a[k-1] * np.cos(k * base_angles)
            r += b[k-1] * np.sin(k * base_angles)
        
        # Enforce positive radius to avoid self-intersection/singularity
        # If r <= 0, the point is at center or invalid. 
        # We can clip to a small epsilon.
        r = np.maximum(r, 0.1)
        
        x = cx + r * np.cos(base_angles)
        y = cy + r * np.sin(base_angles)
        return np.column_stack([x, y])

    def compute_score(points):
        """Compute min triangle area / polygon area (assuming convex)."""
        try:
            pts = points - points.mean(axis=0)
            x = pts[:, 0]
            y = pts[:, 1]
            
            # Shoelace formula for polygon area
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            area_poly = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
            
            if area_poly < 1e-9:
                return 0.0
            
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_area = np.min(areas)
            return min_area / area_poly
        except Exception:
            return 0.0

    def _normalize_points(points):
        """Center and scale points to unit polygon area (assuming convex)."""
        pts = points - points.mean(axis=0)
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        if area < 1e-9:
            return pts
        pts = pts * np.sqrt(1.0 / area)
        return pts

    def soft_objective(params, beta=100.0):
        """
        Objective: minimize -soft_min(triangle areas) for unit polygon area.
        """
        pts = params_to_points(params)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def hard_objective(params):
        """
        Objective: minimize -min(triangle areas) for unit polygon area.
        """
        pts = params_to_points(params)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Differential Evolution bounds
    # Tighter bounds to ensure convexity and numerical stability
    # cx, cy in [-0.2, 0.2]
    # r0 in [0.9, 1.2]
    # a_k, b_k in [-0.1, 0.1]
    bounds = [
        (-0.2, 0.2),  # cx
        (-0.2, 0.2),  # cy
        (0.9, 1.2),   # r0
    ]
    for _ in range(K):
        bounds.append((-0.1, 0.1))  # a_k
        bounds.append((-0.1, 0.1))  # b_k

    best_score = -float('inf')
    best_params = None

    # 1. Differential Evolution
    # Faster objective allows for larger population and more iterations
    try:
        de_res = opt.differential_evolution(
            soft_objective,
            bounds,
            args=(100.0,),
            maxiter=150,
            popsize=30,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        best_params = de_res.x
        best_score = -soft_objective(best_params, 100.0)
    except Exception as e:
        # Fallback to regular polygon
        best_params = np.zeros(n_params)
        best_params[2] = 1.0
        best_score = compute_score(params_to_points(best_params))

    # 2. Refinement using BFGS
    if time.time() < start_time + time_limit - 15:
        try:
            # Increase beta for sharper min approximation
            res = opt.minimize(
                lambda x: soft_objective(x, 300.0),
                best_params,
                method='BFGS',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < soft_objective(best_params, 300.0):
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    # 3. Final Polish with Powell on hard objective (exact min)
    if time.time() < start_time + time_limit - 5:
        try:
            res = opt.minimize(
                hard_objective,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            candidate_params = res.x
            candidate_score = -hard_objective(candidate_params)
            if candidate_score > best_score:
                best_params = candidate_params
                best_score = candidate_score
        except Exception:
            pass

    # Convert best params to points
    best_points = params_to_points(best_params)
    
    # Final check against regular polygon
    reg_poly_params = np.zeros(n_params)
    reg_poly_params[2] = 1.0
    reg_poly_points = params_to_points(reg_poly_params)
    reg_poly_score = compute_score(reg_poly_points)
    
    if reg_poly_score > best_score:
        best_points = reg_poly_points
        best_score = reg_poly_score

    return best_points
