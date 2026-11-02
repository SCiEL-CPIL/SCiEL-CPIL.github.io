"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin (150s limit)
    
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

    def soft_min(values, alpha):
        """Numerically stable soft-min using log-sum-exp."""
        values = np.asarray(values)
        min_val = np.min(values, axis=-1, keepdims=True)
        shifted = values - min_val
        exp_shifted = np.exp(-alpha * shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        log_sum = np.log(np.maximum(sum_exp, 1e-300))
        return min_val - log_sum / alpha

    def smooth_objective(params, alpha=50.0):
        """Smooth objective using soft-min for all constraints."""
        x = params[0::2]
        y = params[1::2]
        
        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        soft_boundary = soft_min(boundary_terms, alpha)
        
        # Soft-min of pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)
        soft_pairwise = soft_min(dist_matrix, alpha)
        
        # Combine: radius_i = min(soft_boundary_i, soft_pairwise_i / 2)
        # Use another soft-min to combine these two
        combined_terms = np.stack([soft_boundary, soft_pairwise / 2.0], axis=1)
        soft_radii = soft_min(combined_terms, alpha)
        
        return -np.sum(soft_radii)

    def objective(params):
        """Hard objective (exact)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def grid_init(jitter=0.0):
        """Generate grid-based initial positions."""
        side = 6  # 6x6 grid for 32 circles
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                if jitter > 0:
                    x0[idx] += np.random.randn() * jitter
                    y0[idx] += np.random.randn() * jitter
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        x0 = np.clip(x0, 0.01, 0.99)
        y0 = np.clip(y0, 0.01, 0.99)
        return x0, y0

    def hex_init(jitter=0.0):
        """Generate hexagonal lattice initial positions."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        rows = 6
        cols = 6
        dx = 1.0 / (cols + 0.5)
        dy = 1.0 / (rows + 0.5)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x0[idx] = (j + 0.5 + (0.5 if i % 2 == 1 else 0)) * dx
                y0[idx] = (i + 0.5) * dy
                if jitter > 0:
                    x0[idx] += np.random.randn() * jitter
                    y0[idx] += np.random.randn() * jitter
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        x0 = np.clip(x0, 0.01, 0.99)
        y0 = np.clip(y0, 0.01, 0.99)
        return x0, y0

    def random_init():
        """Generate random initial positions with minimum spacing."""
        x0 = np.random.uniform(0.05, 0.95, N)
        y0 = np.random.uniform(0.05, 0.95, N)
        return x0, y0

    def voronoi_init():
        """Generate Voronoi-like initial positions using farthest-point sampling."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        x0[0] = 0.5
        y0[0] = 0.5
        for i in range(1, N):
            if time.time() > deadline - 30:
                break
            best_pos = (0.5, 0.5)
            best_min_dist = -1
            for _ in range(200):
                cand_x = np.random.uniform(0.05, 0.95)
                cand_y = np.random.uniform(0.05, 0.95)
                dists = np.sqrt((x0[:i] - cand_x)**2 + (y0[:i] - cand_y)**2)
                min_dist = np.min(dists) if i > 0 else 1.0
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_pos = (cand_x, cand_y)
            x0[i] = best_pos[0]
            y0[i] = best_pos[1]
        return x0, y0

    def run_powell(params_init, maxiter=5000, xtol=1e-12, ftol=1e-12, use_smooth=False, alpha=50.0):
        """Run Powell optimization with time check."""
        if time.time() > deadline - 15:
            return None, None
        
        if use_smooth:
            obj_func = lambda p: smooth_objective(p, alpha)
        else:
            obj_func = objective
        
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                obj_func,
                params_init,
                method='Powell',
                bounds=bounds,
                options={'maxiter': maxiter, 'xtol': xtol, 'ftol': ftol}
            )
            return res.x, res.fun
        except Exception:
            return None, None

    def local_refinement(x, y, max_rounds=100):
        """Local refinement with adaptive step sizes and more directions."""
        radii = compute_radii(x, y)
        sum_r = np.sum(radii)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(N):
                # Adaptive step size based on current radius
                base_step = max(radii[i] * 0.05, 0.0005)
                step_sizes = [base_step, base_step * 0.5, base_step * 0.25, base_step * 0.1, base_step * 2]
                
                for step_size in step_sizes:
                    # Try 16 directions
                    for angle_idx in range(16):
                        angle = 2 * np.pi * angle_idx / 16
                        dx_try = step_size * np.cos(angle)
                        dy_try = step_size * np.sin(angle)
                        
                        x_try = x.copy()
                        y_try = y.copy()
                        x_try[i] += dx_try
                        y_try[i] += dy_try
                        
                        if x_try[i] < 0.001 or x_try[i] > 0.999 or y_try[i] < 0.001 or y_try[i] > 0.999:
                            continue
                        
                        radii_try = compute_radii(x_try, y_try)
                        sum_try = np.sum(radii_try)
                        
                        if sum_try > sum_r + 1e-14:
                            x = x_try
                            y = y_try
                            radii = radii_try
                            sum_r = sum_try
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return x, y

    def simulated_annealing(x, y, max_steps=20000, t0=0.01, t1=0.0001):
        """Simulated annealing polish."""
        radii = compute_radii(x, y)
        sum_r = np.sum(radii)
        best_x = x.copy()
        best_y = y.copy()
        best_sum = sum_r
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Temperature schedule
            progress = step / max_steps
            temp = t0 * (t1 / t0) ** progress
            
            # Random move
            i = np.random.randint(N)
            angle = np.random.uniform(0, 2 * np.pi)
            step_size = np.random.uniform(0.001, 0.02) * (1.0 - 0.5 * progress)
            dx_try = step_size * np.cos(angle)
            dy_try = step_size * np.sin(angle)
            
            x_try = x.copy()
            y_try = y.copy()
            x_try[i] += dx_try
            y_try[i] += dy_try
            
            if x_try[i] < 0.001 or x_try[i] > 0.999 or y_try[i] < 0.001 or y_try[i] > 0.999:
                continue
            
            radii_try = compute_radii(x_try, y_try)
            sum_try = np.sum(radii_try)
            
            delta = sum_try - sum_r
            if delta > 0 or np.random.rand() < np.exp(delta / max(temp, 1e-10)):
                x = x_try
                y = y_try
                radii = radii_try
                sum_r = sum_try
                
                if sum_r > best_sum:
                    best_sum = sum_r
                    best_x = x.copy()
                    best_y = y.copy()
        
        return best_x, best_y

    # Phase 1: Multiple diverse initializations
    np.random.seed(42)
    initializations = []
    
    # Grid
    x0, y0 = grid_init(jitter=0.0)
    initializations.append((x0, y0))
    
    # Hexagonal
    x0, y0 = hex_init(jitter=0.0)
    initializations.append((x0, y0))
    
    # Random
    x0, y0 = random_init()
    initializations.append((x0, y0))
    
    # Voronoi-like
    x0, y0 = voronoi_init()
    initializations.append((x0, y0))
    
    # Jittered grid
    x0, y0 = grid_init(jitter=0.02)
    initializations.append((x0, y0))
    
    # Jittered hex
    x0, y0 = hex_init(jitter=0.02)
    initializations.append((x0, y0))
    
    best_params = None
    best_obj = np.inf
    
    for init_idx, (x_init, y_init) in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        params_init = np.zeros(2 * N)
        params_init[0::2] = x_init
        params_init[1::2] = y_init
        
        # Smooth pass
        res_x, res_fun = run_powell(params_init, maxiter=2000, xtol=1e-10, ftol=1e-10, use_smooth=True, alpha=40.0)
        if res_x is None:
            continue
        
        hard_fun = objective(res_x)
        
        # Hard pass
        res_x2, res_fun2 = run_powell(res_x, maxiter=3000, xtol=1e-12, ftol=1e-12, use_smooth=False)
        if res_x2 is not None:
            hard_fun2 = objective(res_x2)
            if hard_fun2 < hard_fun:
                res_x = res_x2
                hard_fun = hard_fun2
        
        if hard_fun < best_obj:
            best_obj = hard_fun
            best_params = res_x.copy()
    
    if best_params is None:
        # Fallback
        x0, y0 = grid_init(jitter=0.0)
        best_params = np.zeros(2 * N)
        best_params[0::2] = x0
        best_params[1::2] = y0
        best_obj = objective(best_params)
    
    # Phase 2: Local refinement
    if time.time() < deadline - 20:
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        x_best, y_best = local_refinement(x_best, y_best, max_rounds=150)
        
        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best
        best_obj = objective(best_params)
    
    # Phase 3: Simulated annealing polish
    if time.time() < deadline - 10:
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        x_best, y_best = simulated_annealing(x_best, y_best, max_steps=15000)
        
        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best
        best_obj = objective(best_params)
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        res_x, res_fun = run_powell(best_params, maxiter=10000, xtol=1e-14, ftol=1e-14, use_smooth=False)
        if res_x is not None and res_fun < best_obj:
            best_params = res_x.copy()
            best_obj = res_fun
    
    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)
    
    # Shrink radii slightly to ensure strict disjointness (floating point safety)
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
