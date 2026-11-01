"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin (150s limit)

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

    def phyllotaxis_init():
        """Generate phyllotaxis initialization - low-discrepancy distribution."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(N):
            r = 0.45 * np.sqrt((i + 0.5) / N)
            theta = golden_angle * i
            x0[i] = 0.5 + r * np.cos(theta)
            y0[i] = 0.5 + r * np.sin(theta)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Generate hexagonal lattice initialization."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        rows = 6
        cols_per_row = 6
        spacing = 1.0 / (cols_per_row + 0.5)
        
        idx = 0
        for i in range(rows):
            n_cols = cols_per_row if i % 2 == 0 else cols_per_row - 1
            offset = spacing / 2 if i % 2 == 1 else 0
            for j in range(n_cols):
                if idx >= N:
                    break
                x0[idx] = (j + 0.5) * spacing + offset
                y0[idx] = (i + 0.5) * (1.0 / rows)
                idx += 1
        
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def grid_init():
        """Generate grid initialization."""
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

    def random_init(seed_val):
        """Generate random initialization."""
        rng = np.random.RandomState(seed_val)
        x0 = rng.uniform(0.05, 0.95, N)
        y0 = rng.uniform(0.05, 0.95, N)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def voronoi_refine(params, n_iter=50):
        """Voronoi-based refinement: move each center toward centroid of its Voronoi cell."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        
        for _ in range(n_iter):
            if time.time() > deadline - 15:
                break
            
            grid_size = 50
            gx, gy = np.meshgrid(np.linspace(0.01, 0.99, grid_size),
                                  np.linspace(0.01, 0.99, grid_size))
            gx_flat = gx.flatten()
            gy_flat = gy.flatten()
            
            dists = np.zeros((len(gx_flat), N))
            for i in range(N):
                dists[:, i] = (gx_flat - x[i])**2 + (gy_flat - y[i])**2
            nearest = np.argmin(dists, axis=1)
            
            new_x = np.zeros(N)
            new_y = np.zeros(N)
            counts = np.zeros(N)
            for i in range(N):
                mask = (nearest == i)
                if np.sum(mask) > 0:
                    new_x[i] = np.mean(gx_flat[mask])
                    new_y[i] = np.mean(gy_flat[mask])
                    counts[i] = np.sum(mask)
                else:
                    new_x[i] = x[i]
                    new_y[i] = y[i]
            
            alpha_v = 0.3
            x_new = x + alpha_v * (new_x - x)
            y_new = y + alpha_v * (new_y - y)
            
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)
            
            obj_old = -np.sum(compute_radii(x, y))
            obj_new = -np.sum(compute_radii(x_new, y_new))
            
            if obj_new < obj_old:
                x = x_new
                y = y_new
            else:
                break
        
        params_out = np.zeros(2 * N)
        params_out[0::2] = x
        params_out[1::2] = y
        return params_out

    # Radius-explicit formulation from inspiration 1
    def radius_objective(vars):
        """Objective with explicit radii and penalty for violations."""
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        obj = -np.sum(R)
        pen = 0.0
        
        # Pairwise overlap penalty
        dx = X[:, np.newaxis] - X[np.newaxis, :]
        dy = Y[:, np.newaxis] - Y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        overlap = R[:, np.newaxis] + R[np.newaxis, :] - d
        overlap = np.maximum(overlap, 0)
        pen += np.sum(overlap**2)
        
        # Boundary penalty
        pen += np.sum(np.maximum(X - R, 0)**2)  # x - r < 0
        pen += np.sum(np.maximum(1.0 - X - R, 0)**2)  # 1 - x - r < 0
        pen += np.sum(np.maximum(Y - R, 0)**2)
        pen += np.sum(np.maximum(1.0 - Y - R, 0)**2)
        
        weight = 100.0
        return obj + weight * pen

    def radius_objective_grad(vars):
        """Gradient of radius objective."""
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        weight = 100.0
        
        # Pairwise overlap gradient
        dx = X[:, np.newaxis] - X[np.newaxis, :]
        dy = Y[:, np.newaxis] - Y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1e-10)
        
        overlap = R[:, np.newaxis] + R[np.newaxis, :] - d
        overlap = np.maximum(overlap, 0)
        
        # Gradient contribution from pairwise terms
        for i in range(N):
            for j in range(i + 1, N):
                if overlap[i, j] > 0:
                    factor = 2.0 * overlap[i, j]
                    grad[3*i] += weight * factor * (-dx[i, j] / d[i, j])
                    grad[3*i+1] += weight * factor * (-dy[i, j] / d[i, j])
                    grad[3*j] += weight * factor * (dx[i, j] / d[i, j])
                    grad[3*j+1] += weight * factor * (dy[i, j] / d[i, j])
                    grad[3*i+2] += weight * factor
                    grad[3*j+2] += weight * factor
        
        # Boundary gradient
        for i in range(N):
            val = X[i] - R[i]
            if val < 0:
                grad[3*i] += weight * 2.0 * val
                grad[3*i+2] += weight * (-2.0 * val)
            
            val = 1.0 - X[i] - R[i]
            if val < 0:
                grad[3*i] += weight * (-2.0 * val)
                grad[3*i+2] += weight * (-2.0 * val)
            
            val = Y[i] - R[i]
            if val < 0:
                grad[3*i+1] += weight * 2.0 * val
                grad[3*i+2] += weight * (-2.0 * val)
            
            val = 1.0 - Y[i] - R[i]
            if val < 0:
                grad[3*i+1] += weight * (-2.0 * val)
                grad[3*i+2] += weight * (-2.0 * val)
        
        return grad

    def center_to_radius_vars(params):
        """Convert center params to radius-explicit vars."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y) * 0.95  # slight shrink for safety
        vars_out = np.zeros(3 * N)
        vars_out[0::3] = x
        vars_out[1::3] = y
        vars_out[2::3] = radii
        return vars_out

    def radius_vars_to_result(vars):
        """Convert radius vars to result array."""
        X = vars[0::3]
        Y = vars[1::3]
        R = vars[2::3]
        return np.column_stack((X, Y, R))

    bounds_center = [(0.0, 1.0)] * (2 * N)
    bounds_radius = []
    for i in range(N):
        bounds_radius.append((0.0, 1.0))
        bounds_radius.append((0.0, 1.0))
        bounds_radius.append((1e-5, 0.5))

    # Generate multiple initializations
    inits = []
    inits.append(('phyllotaxis', phyllotaxis_init()))
    inits.append(('hex', hex_init()))
    inits.append(('grid', grid_init()))
    for s in [1, 7, 13, 42]:
        inits.append((f'rand_{s}', random_init(s)))

    best_params = None
    best_obj = np.inf
    best_radius_vars = None
    best_radius_obj = np.inf

    # Phase 1: Optimize each initialization with Powell (center-based)
    for name, params0 in inits:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                objective, params0, method='Powell', bounds=bounds_center,
                options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = phyllotaxis_init()
        best_obj = objective(best_params)

    # Phase 2: Voronoi refinement
    if time.time() < deadline - 40:
        try:
            voronoi_params = voronoi_refine(best_params, n_iter=30)
            voronoi_obj = objective(voronoi_params)
            if voronoi_obj < best_obj:
                best_obj = voronoi_obj
                best_params = voronoi_params
        except Exception:
            pass

    # Phase 3: Smooth objective optimization with L-BFGS-B from best
    if time.time() < deadline - 30:
        try:
            res = minimize(
                smooth_objective, best_params, method='L-BFGS-B', bounds=bounds_center,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_obj = true_obj
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Powell refinement from best smooth result
    if time.time() < deadline - 25:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds_center,
                options={'maxiter': 4000, 'xtol': 1e-13, 'ftol': 1e-13}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Radius-explicit optimization (inspiration 1 approach)
    if time.time() < deadline - 15:
        try:
            # Convert best center solution to radius vars
            r_vars0 = center_to_radius_vars(best_params)
            
            res_r = minimize(
                radius_objective, r_vars0, jac=radius_objective_grad,
                method='L-BFGS-B', bounds=bounds_radius,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate with true radii
            X_r = res_r.x[0::3]
            Y_r = res_r.x[1::3]
            true_sum = np.sum(compute_radii(X_r, Y_r))
            
            if true_sum > -best_obj:
                best_obj = -true_sum
                best_params = np.zeros(2 * N)
                best_params[0::2] = X_r
                best_params[1::2] = Y_r
                best_radius_vars = res_r.x.copy()
        except Exception:
            pass

    # Phase 6: Additional smooth + Powell cycle
    if time.time() < deadline - 10:
        try:
            res = minimize(
                smooth_objective, best_params, method='L-BFGS-B', bounds=bounds_center,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_obj = true_obj
                best_params = res.x.copy()
                
                # Quick Powell polish
                res2 = minimize(
                    objective, best_params, method='Powell', bounds=bounds_center,
                    options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-14}
                )
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 7: Local perturbation refinement
    if time.time() < deadline - 5:
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        sum_best = np.sum(compute_radii(x_best, y_best))

        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                for step_size in [0.003, 0.0015, 0.0008, 0.0004, 0.0002]:
                    for dx_try in [-step_size, 0, step_size]:
                        for dy_try in [-step_size, 0, step_size]:
                            if dx_try == 0 and dy_try == 0:
                                continue
                            x_try = x_best.copy()
                            y_try = y_best.copy()
                            x_try[i] += dx_try
                            y_try[i] += dy_try

                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue

                            sum_try = np.sum(compute_radii(x_try, y_try))

                            if sum_try > sum_best + 1e-15:
                                x_best = x_try
                                y_best = y_try
                                sum_best = sum_try
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly to ensure strict disjointness (floating point safety)
    radii *= 0.99999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
