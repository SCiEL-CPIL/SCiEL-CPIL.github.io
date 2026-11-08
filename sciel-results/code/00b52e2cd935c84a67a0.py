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

    # Fourier modes: K=3 is usually sufficient for small deviations from regular polygon.
    # K=4 adds flexibility but increases search space. Let's stick to K=3 or 4.
    # The previous program used K=4. Let's use K=3 to keep the search space smaller and focus on the most relevant modes, 
    # or K=4 if we think higher harmonics matter. For 13 points, modes 1, 2, 3, 4 are distinct.
    # Let's use K=3 to reduce dimensionality (2 + 1 + 6 = 9 params vs 11).
    # Actually, let's keep K=4 as in the original, it's only 2 extra params.
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
        
        # Ensure positive radius. 
        # With tight bounds, this shouldn't trigger, but good for safety.
        # Using a soft floor or just clipping. Clipping is non-differentiable.
        # To keep it differentiable for BFGS, we should ensure bounds prevent r<=0.
        # If r0 >= 0.8 and sum |a_k|, |b_k| <= 0.3, r >= 0.5.
        # So we don't need the max if bounds are tight.
        x = cx + r * np.cos(base_angles)
        y = cy + r * np.sin(base_angles)
        return np.column_stack([x, y])

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def soft_objective(params, beta=100.0):
        """
        Objective: minimize -soft_min(triangle areas) for unit hull area.
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
        # To avoid overflow, subtract max
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def hard_objective(params):
        """
        Objective: minimize -min(triangle areas) for unit hull area.
        """
        pts = params_to_points(params)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Tighter bounds to ensure r > 0 and smoothness
    # r0 in [0.8, 1.2]
    # a_k, b_k in [-0.3, 0.3]
    # cx, cy in [-0.2, 0.2]
    bounds = [
        (-0.2, 0.2),  # cx
        (-0.2, 0.2),  # cy
        (0.8, 1.2),   # r0
    ]
    for _ in range(K):
        bounds.append((-0.3, 0.3))  # a_k
        bounds.append((-0.3, 0.3))  # b_k

    best_score = -float('inf')
    best_params = None

    # 1. Differential Evolution
    # Use a good population size. 13 points, 11 params. Popsize 20-30 is reasonable.
    try:
        de_res = opt.differential_evolution(
            soft_objective,
            bounds,
            args=(100.0,),
            maxiter=100,
            popsize=25,
            tol=1e-5,
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

    # 2. Refinement using L-BFGS-B
    # L-BFGS-B is better than BFGS for bounded problems and large scale (though this is small).
    if time.time() < start_time + time_limit - 20:
        try:
            # Use a higher beta for sharper min in refinement
            res = opt.minimize(
                lambda x: soft_objective(x, 500.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.fun < soft_objective(best_params, 500.0):
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    # 3. Multi-start local optimization from best_params and regular polygon
    # Sometimes DE finds a good local min, but the global min is slightly different.
    # Let's try optimizing from the regular polygon too, just in case.
    if time.time() < start_time + time_limit - 10:
        starts = [best_params.copy()]
        # Add regular polygon as a start point
        reg_params = np.zeros(n_params)
        reg_params[2] = 1.0
        starts.append(reg_params)
        
        # Add a few random perturbations around best_params
        for _ in range(3):
            perturbed = best_params + np.random.normal(0, 0.01, n_params)
            # Clip to bounds
            for i in range(n_params):
                perturbed[i] = np.clip(perturbed[i], bounds[i][0], bounds[i][1])
            starts.append(perturbed)

        for s in starts:
            if time.time() >= start_time + time_limit - 5:
                break
            try:
                res = opt.minimize(
                    lambda x: soft_objective(x, 500.0),
                    s,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-8}
                )
                if res.fun < soft_objective(s, 500.0):
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_params = res.x
            except Exception:
                pass

    # 4. Final Polish with Powell on hard objective
    if time.time() < start_time + time_limit - 2:
        try:
            res = opt.minimize(
                hard_objective,
                best_params,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-12, 'ftol': 1e-12}
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
