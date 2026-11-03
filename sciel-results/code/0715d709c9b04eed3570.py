import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def bary_to_cart(uv):
    """Convert barycentric coordinates (u, v) to Cartesian.
    u = lambda_1, v = lambda_2, lambda_0 = 1 - u - v
    x = u * 1 + v * 0.5
    y = v * sqrt(3)/2
    """
    u = uv[:, 0]
    v = uv[:, 1]
    x = u + 0.5 * v
    y = v * np.sqrt(3) / 2
    return np.column_stack((x, y))

def min_area_cart(pts):
    """Calculate the minimum area of any triangle formed by the points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det(pj - pi, pk - pi)|
                cross = (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - \
                        (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1])
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a

def negative_min_area(uv_flat):
    """Objective function for minimization: -min_area."""
    uv = uv_flat.reshape(N, 2)
    # Ensure points are inside triangle: u >= 0, v >= 0, u+v <= 1
    # The bounds in differential_evolution handle this.
    pts = bary_to_cart(uv)
    return -min_area_cart(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Define bounds for barycentric coordinates u, v
    # u in [0, 1], v in [0, 1], but we need u+v <= 1.
    # Differential evolution doesn't support linear inequality constraints directly.
    # We can use a penalty or transform variables.
    # Better: use simple bounds [0,1] for u and v, and penalize if u+v > 1.
    # Or use a transformation that guarantees u+v <= 1.
    # Let u = s * cos^2(theta), v = s * sin^2(theta) ... no, that's for disk.
    # For triangle: u = w1, v = w2.
    # We can just use bounds [0,1] and add a large penalty if u+v > 1 - eps.
    # Or, use the fact that we can parameterize the triangle with two variables in [0,1] via a mapping.
    # Mapping: x = u, y = v * (1-u). Then u in [0,1], v in [0,1].
    # Let's stick to u,v in [0,1] and penalty.
    
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N

    def penalty_min_area(uv_flat):
        uv = uv_flat.reshape(N, 2)
        # Check if any point is outside
        # u >= 0, v >= 0 is handled by bounds.
        # Check u + v <= 1
        violations = np.sum(np.maximum(0, uv[:, 0] + uv[:, 1] - 1.0))
        if violations > 1e-6:
            return 1e6 * violations  # Large penalty
        pts = bary_to_cart(uv)
        return -min_area_cart(pts)

    # 1. Global Search with Differential Evolution
    # maxiter and popsize need to be tuned for time limit.
    # N=11, 22 variables.
    # DE can be slow. Let's try a moderate setting.
    best_result = None
    best_val = -float('inf')
    
    # Initial guess: random points
    rng = np.random.RandomState(42)
    
    # Run DE
    try:
        res = differential_evolution(
            penalty_min_area, 
            bounds=bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=0.01,
            polish=False, # We will polish separately
            init='sobol'
        )
        if time.time() < deadline:
            best_val = -res.fun
            best_result = res.x
    except Exception:
        pass

    # If DE didn't finish or failed, use a few random restarts
    if best_result is None:
        for _ in range(5):
            uv_init = rng.random((N, 2))
            # Project to triangle if needed? DE bounds ensure u,v in [0,1].
            # We need u+v <= 1.
            # Simple projection: if u+v > 1, scale down.
            s = uv_init[:, 0] + uv_init[:, 1]
            mask = s > 1.0
            if np.any(mask):
                uv_init[mask] /= s[mask, None]
            best_result = uv_init.flatten()
            best_val = -penalty_min_area(best_result)

    # 2. Local Polish with L-BFGS-B
    # We need to handle the constraint u+v <= 1.
    # L-BFGS-B supports bounds but not linear inequalities.
    # We can use the penalty approach in the objective function for local opt too,
    # or use SLSQP which supports constraints.
    # SLSQP might be slower but handles constraints.
    # Let's try L-BFGS-B with penalty.
    
    def local_obj(uv_flat):
        uv = uv_flat.reshape(N, 2)
        violations = np.sum(np.maximum(0, uv[:, 0] + uv[:, 1] - 1.0))
        if violations > 1e-8:
            return 1e4 * violations
        pts = bary_to_cart(uv)
        # Use a smooth approximation? No, min is non-smooth.
        # L-BFGS-B works okay with non-smooth functions if we are careful, 
        # but it might get stuck.
        # Let's just use the raw min area.
        return -min_area_cart(pts)

    # Try polishing from the DE result
    if time.time() < deadline - 5:
        try:
            res_local = minimize(
                local_obj, 
                best_result, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if -res_local.fun > best_val:
                best_val = -res_local.fun
                best_result = res_local.x
        except Exception:
            pass

    # If time remains, try a few more local restarts from random points
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            uv_init = rng.random((N, 2))
            s = uv_init[:, 0] + uv_init[:, 1]
            mask = s > 1.0
            if np.any(mask):
                uv_init[mask] /= s[mask, None]
            
            try:
                res_local = minimize(
                    local_obj, 
                    uv_init.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                if -res_local.fun > best_val:
                    best_val = -res_local.fun
                    best_result = res_local.x
            except Exception:
                pass

    # Convert best barycentric coordinates to Cartesian
    uv_best = best_result.reshape(N, 2)
    # Ensure strict interior to avoid boundary issues if any
    # The evaluator likely checks "on or inside".
    pts = bary_to_cart(uv_best)
    
    return pts
