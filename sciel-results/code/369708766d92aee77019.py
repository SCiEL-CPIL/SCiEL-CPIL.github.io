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

    def gradient(params):
        """Compute gradient of the objective (negative sum of radii)."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        n = len(x)
        
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        
        grad = np.zeros(2 * n)
        
        for i in range(n):
            r_i = radii[i]
            if r_i <= 0:
                continue
            
            x_min = x[i]
            x_max = 1 - x[i]
            y_min = y[i]
            y_max = 1 - y[i]
            min_b = min(x_min, x_max, y_min, y_max)
            
            min_d = min_dist[i]
            nearest_j = np.argmin(dist_matrix[i])
            
            if min_b < min_d / 2.0:
                if x_min == min_b:
                    grad[2*i] = -1.0
                elif x_max == min_b:
                    grad[2*i] = 1.0
                elif y_min == min_b:
                    grad[2*i+1] = -1.0
                else:
                    grad[2*i+1] = 1.0
            else:
                d_ij = dist_matrix[i, nearest_j]
                if d_ij > 1e-15:
                    grad[2*i] = (x[i] - x[nearest_j]) / (2 * d_ij)
                    grad[2*i+1] = (y[i] - y[nearest_j]) / (2 * d_ij)
        
        return grad

    def smooth_objective(params, alpha=80.0):
        """Smooth objective using soft-min via log-sum-exp."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

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
        side = int(np.ceil(np.sqrt(N)))
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

    def phyllotaxis_init():
        """Generate phyllotaxis (sunflower) initialization."""
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

    def voronoi_refine(params, n_iter=50):
        """Voronoi-based refinement."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        
        for _ in range(n_iter):
            if time.time() > deadline - 20:
                break
            
            grid_size = 80
            gx, gy = np.meshgrid(np.linspace(0.005, 0.995, grid_size),
                                  np.linspace(0.005, 0.995, grid_size))
            gx_flat = gx.flatten()
            gy_flat = gy.flatten()
            
            dists = np.zeros((len(gx_flat), N))
            for i in range(N):
                dists[:, i] = (gx_flat - x[i])**2 + (gy_flat - y[i])**2
            nearest = np.argmin(dists, axis=1)
            
            new_x = np.zeros(N)
            new_y = np.zeros(N)
            for i in range(N):
                mask = (nearest == i)
                if np.sum(mask) > 0:
                    new_x[i] = np.mean(gx_flat[mask])
                    new_y[i] = np.mean(gy_flat[mask])
                else:
                    new_x[i] = x[i]
                    new_y[i] = y[i]
            
            alpha_v = 0.3
            x_new = x + alpha_v * (new_x - x)
            y_new = y + alpha_v * (new_y - y)
            
            x_new = np.clip(x_new, 0.005, 0.995)
            y_new = np.clip(y_new, 0.005, 0.995)
            
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

    def local_perturbation(params, n_rounds=300):
        """Fine local perturbation refinement."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        sum_best = np.sum(compute_radii(x, y))
        
        for _ in range(n_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for step_size in [0.003, 0.0015, 0.0007, 0.0003, 0.0001]:
                    for dx_try in [-step_size, 0, step_size]:
                        for dy_try in [-step_size, 0, step_size]:
                            if dx_try == 0 and dy_try == 0:
                                continue
                            x_try = x.copy()
                            y_try = y.copy()
                            x_try[i] += dx_try
                            y_try[i] += dy_try

                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue

                            sum_try = np.sum(compute_radii(x_try, y_try))

                            if sum_try > sum_best + 1e-14:
                                x = x_try
                                y = y_try
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

        params_out = np.zeros(2 * N)
        params_out[0::2] = x
        params_out[1::2] = y
        return params_out

    def annealed_search(params, n_iter=500):
        """Simulated annealing style search with random perturbations."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        current_sum = np.sum(compute_radii(x, y))
        best_sum = current_sum
        best_x = x.copy()
        best_y = y.copy()
        
        rng = np.random.RandomState(999)
        T0 = 0.02
        T_end = 0.001
        cooling = (T_end / T0) ** (1.0 / max(n_iter - 1, 1))
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                break
            T = T0 * (cooling ** it)
            
            # Pick a random circle to perturb
            i = rng.randint(N)
            perturbation_scale = T * 5
            dx = rng.uniform(-perturbation_scale, perturbation_scale)
            dy = rng.uniform(-perturbation_scale, perturbation_scale)
            
            x_try = x.copy()
            y_try = y.copy()
            x_try[i] += dx
            y_try[i] += dy
            
            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                continue
            
            sum_try = np.sum(compute_radii(x_try, y_try))
            delta = sum_try - current_sum
            
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_try
                y = y_try
                current_sum = sum_try
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x = x.copy()
                    best_y = y.copy()
        
        params_out = np.zeros(2 * N)
        params_out[0::2] = best_x
        params_out[1::2] = best_y
        return params_out

    bounds = [(0.0, 1.0)] * (2 * N)

    # Generate multiple initializations
    inits = []
    inits.append(grid_init())
    inits.append(hex_init())
    inits.append(phyllotaxis_init())
    for s in [1, 7, 13, 42, 123, 777, 2024, 5555, 8888, 12345]:
        inits.append(random_init(s))

    best_params = None
    best_obj = np.inf

    # Phase 1: Optimize each initialization with Powell
    for idx, params0 in enumerate(inits):
        if time.time() > deadline - 55:
            break
        try:
            res = minimize(
                objective, params0, method='Powell', bounds=bounds,
                options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = grid_init()
        best_obj = objective(best_params)

    # Phase 2: Voronoi refinement
    if time.time() < deadline - 40:
        try:
            voronoi_params = voronoi_refine(best_params, n_iter=40)
            voronoi_obj = objective(voronoi_params)
            if voronoi_obj < best_obj:
                best_obj = voronoi_obj
                best_params = voronoi_params
        except Exception:
            pass

    # Phase 3: Gradient-based optimization (L-BFGS-B)
    if time.time() < deadline - 35:
        try:
            res = minimize(
                objective, best_params, method='L-BFGS-B', bounds=bounds,
                jac=gradient,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_obj = true_obj
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Smooth objective optimization
    if time.time() < deadline - 30:
        try:
            res = minimize(
                smooth_objective, best_params, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_obj = true_obj
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 5: Powell refinement
    if time.time() < deadline - 25:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 8000, 'xtol': 1e-13, 'ftol': 1e-13}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 6: Annealed search to escape local minima
    if time.time() < deadline - 20:
        try:
            annealed_params = annealed_search(best_params, n_iter=800)
            annealed_obj = objective(annealed_params)
            if annealed_obj < best_obj:
                best_obj = annealed_obj
                best_params = annealed_params
        except Exception:
            pass

    # Phase 7: Another round of smooth + Powell
    if time.time() < deadline - 15:
        try:
            res = minimize(
                smooth_objective, best_params, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_obj = true_obj
                best_params = res.x.copy()
                
            res2 = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 8: Voronoi again from improved position
    if time.time() < deadline - 12:
        try:
            voronoi_params2 = voronoi_refine(best_params, n_iter=30)
            voronoi_obj2 = objective(voronoi_params2)
            if voronoi_obj2 < best_obj:
                best_obj = voronoi_obj2
                best_params = voronoi_params2
        except Exception:
            pass

    # Phase 9: Final local perturbation
    if time.time() < deadline - 5:
        try:
            local_params = local_perturbation(best_params, n_rounds=500)
            local_obj = objective(local_params)
            if local_obj < best_obj:
                best_obj = local_obj
                best_params = local_params
        except Exception:
            pass

    # Phase 10: Final tight polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 10000, 'xtol': 1e-15, 'ftol': 1e-15}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly to ensure strict disjointness
    radii *= 0.99999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
