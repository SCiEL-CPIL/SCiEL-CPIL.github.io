"""Hybrid CMA-ES + L-BFGS-B + multi-start refinement for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: Sobol initialization, CMA-ES global search, L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = None
        best_f = float('inf')
        all_x = []
        all_f = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f, all_x, all_f
            if eval_count >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            f_val = float(func(x))
            all_x.append(x.copy())
            all_f.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Sobol-like Initialization ---
        def halton_sequence(count, dim):
            primes = [2, 3, 5, 7, 11]
            points = []
            for i in range(1, count + 1):
                point = []
                for d in range(dim):
                    b = primes[d]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= b
                        r += f * (idx % b)
                        idx //= b
                    point.append(r)
                point = np.array(point) * 10.0 - 5.0
                points.append(point)
            return points

        n_init = 30
        init_points = halton_sequence(n_init, dim)
        init_points.append(np.zeros(dim))
        
        for pt in init_points:
            if eval_count >= budget - 50 or time.time() > deadline:
                break
            wrapped_func(pt)

        # --- Phase 2: CMA-ES ---
        if best_x is None:
            best_x = np.zeros(dim)
        
        # CMA-ES parameters
        pop_size = 10
        sigma = 2.0
        x_mean = best_x.copy()
        C = np.eye(dim)
        step_size = sigma
        
        # Pre-compute constants
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial Cholesky factor
        L = np.linalg.cholesky(C)

        cma_evals = 0
        max_cma_evals = int(budget * 0.6)
        stagnation_counter = 0

        while eval_count < budget - 50 and cma_evals < max_cma_evals and time.time() < deadline:
            # Generate population
            xs = []
            for i in range(pop_size):
                z = np.random.normal(0, 1, dim)
                x = x_mean + step_size * (L @ z)
                x = np.clip(x, -5.0, 5.0)
                xs.append(x)
            
            # Evaluate
            fs = []
            for x in xs:
                f = wrapped_func(x)
                fs.append(f)
                cma_evals += 1
                if eval_count >= budget - 50:
                    break
            
            if len(fs) < pop_size:
                break

            # Sort
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * xs_sorted[i]
            
            # Update step size (sigma)
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (C_inv_sqrt @ (x_new_mean - x_mean)) / step_size
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (cma_evals // pop_size + 1))) < (1 + 2 / (dim + 1)) * chi_n)
            
            # Update C
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / step_size
            
            delta_h = 1 if h_sigma else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(w[i] * np.outer(xs_sorted[i] - x_mean, xs_sorted[i] - x_mean) for i in range(mu))
            
            # Regularization
            C += 1e-10 * np.eye(dim)
            
            # Update step size
            step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure step size doesn't get too small or too large
            step_size = np.clip(step_size, 1e-8, 10.0)
            
            x_mean = x_new_mean
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = 0.5 * C + 0.5 * np.eye(dim)
                L = np.linalg.cholesky(C)

            # If stagnated for too long, restart from best point with larger sigma
            if stagnation_counter > 20:
                x_mean = best_x.copy()
                step_size = 1.0
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_counter = 0
                for _ in range(5):
                    if eval_count >= budget - 50: break
                    pt = best_x + np.random.normal(0, 2.0, dim)
                    pt = np.clip(pt, -5.0, 5.0)
                    wrapped_func(pt)

        # --- Phase 3: Multi-start L-BFGS-B Refinement ---
        # Use remaining budget to polish top candidates with L-BFGS-B
        if best_x is not None and eval_count < budget and time.time() < deadline:
            # Collect top candidates for refinement
            # Use best point and top 4 from all evaluated points
            candidates = [best_x.copy()]
            
            # Add top 4 distinct points from history
            if all_x:
                # Sort by function value
                idx_sorted = np.argsort(all_f)
                for idx in idx_sorted[:5]:
                    if idx == 0:
                        continue
                    pt = all_x[idx].copy()
                    # Check if distinct from existing candidates
                    is_distinct = True
                    for c in candidates:
                        if np.linalg.norm(pt - c) < 0.1:
                            is_distinct = False
                            break
                    if is_distinct:
                        candidates.append(pt)
                        if len(candidates) >= 5:
                            break
            
            # Try L-BFGS-B from each candidate
            bounds = [(-5.0, 5.0)] * dim
            for cand in candidates:
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                
                try:
                    result = minimize(
                        wrapped_func,
                        cand,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfun': 50}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
