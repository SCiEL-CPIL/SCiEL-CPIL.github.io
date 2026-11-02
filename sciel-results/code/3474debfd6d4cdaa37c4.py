"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin (150s limit)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=100.0):
        """Smooth objective using soft-min via log-sum-exp for gradient-based methods."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))  # 6
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Hexagonal close-packing initialization."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        s = 1.0 / np.sqrt(2 * N)
        for row in range(8):
            for col in range(8):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                x0[idx] = np.clip(x0[idx], 0.05, 0.95)
                y0[idx] = np.clip(y0[idx], 0.05, 0.95)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def random_init():
        """Random initialization with some spacing."""
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def jittered_init(base, sigma=0.05):
        """Add Gaussian jitter to a base initialization."""
        p = base.copy()
        p += np.random.randn(2 * N) * sigma
        return np.clip(p, 0.01, 0.99)

    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Try multiple initializations
    np.random.seed(42)
    inits = [grid_init(), hex_init()]
    for _ in range(5):
        inits.append(random_init())
    # Add jittered versions of the best structured inits
    inits.append(jittered_init(grid_init(), 0.03))
    inits.append(jittered_init(hex_init(), 0.03))

    for init_params in inits:
        if time.time() > deadline - 60:
            break
        obj_val = objective(init_params)
        if obj_val < best_obj:
            best_params = init_params.copy()
            best_obj = obj_val

    # Multi-stage optimization with increasing alpha (smoothing sharpness)
    alphas = [15.0, 40.0, 80.0, 150.0, 300.0]
    
    for i, alpha in enumerate(alphas):
        if time.time() > deadline - 25:
            break
            
        # Number of restarts decreases as we refine
        n_restarts = 3 if i < 2 else 2
        
        for restart in range(n_restarts):
            if time.time() > deadline - 25:
                break
            
            if restart == 0:
                params_start = best_params.copy()
            elif restart == 1:
                params_start = jittered_init(best_params, 0.02)
            else:
                params_start = jittered_init(best_params, 0.005)
            
            try:
                ftol = 1e-13 if i >= 3 else 1e-11
                res = minimize(
                    lambda p, a=alpha: smooth_objective(p, alpha=a), 
                    params_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': ftol, 'gtol': 1e-10}
                )
                true_obj = objective(res.x)
                if true_obj < best_obj:
                    best_params = res.x
                    best_obj = true_obj
            except Exception:
                pass

    # Phase 3: Powell refinement - multiple passes with tightening tolerances
    powell_configs = [
        {'maxiter': 8000, 'xtol': 1e-12, 'ftol': 1e-12},
        {'maxiter': 15000, 'xtol': 1e-14, 'ftol': 1e-14},
        {'maxiter': 20000, 'xtol': 1e-15, 'ftol': 1e-15},
    ]
    
    for cfg in powell_configs:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options=cfg
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: Final local coordinate-descent polish
    if time.time() < deadline - 3:
        try:
            x = best_params[0::2].copy()
            y = best_params[1::2].copy()
            
            # Simple iterative local improvement: for each circle, try small moves
            # in x and y to see if it improves the total radius sum
            for iteration in range(50):
                if time.time() > deadline - 3:
                    break
                improved = False
                for i in range(N):
                    # Try small perturbations in x
                    for dx in [-0.001, 0.001]:
                        x_new = x.copy()
                        x_new[i] += dx
                        if 0 <= x_new[i] <= 1:
                            r_old = compute_radii(x, y)
                            r_new = compute_radii(x_new, y)
                            if np.sum(r_new) > np.sum(r_old) + 1e-14:
                                x = x_new
                                improved = True
                                break
                    # Try small perturbations in y
                    for dy in [-0.001, 0.001]:
                        y_new = y.copy()
                        y_new[i] += dy
                        if 0 <= y_new[i] <= 1:
                            r_old = compute_radii(x, y)
                            r_new = compute_radii(x, y_new)
                            if np.sum(r_new) > np.sum(r_old) + 1e-14:
                                y = y_new
                                improved = True
                                break
                if not improved:
                    break
            
            final_params = np.zeros(2 * N)
            final_params[0::2] = x
            final_params[1::2] = y
            final_obj = objective(final_params)
            if final_obj < best_obj:
                best_params = final_params
                best_obj = final_obj
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
