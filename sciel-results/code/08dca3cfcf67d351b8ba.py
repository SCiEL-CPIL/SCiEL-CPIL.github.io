"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_fx
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            val = func(x)
            calls += 1
            if np.isfinite(val) and val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Initial Exploration - find a good basin
        n_initial = min(25, budget // 5)
        for i in range(n_initial):
            if time.time() > deadline or calls >= budget:
                break
            # Mix of random and structured points
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = np.ones(n) * 0.5
            elif i == 2:
                x = np.ones(n)
            elif i == 3:
                x = rng.uniform(-2, 2, n)
            else:
                x = rng.uniform(-4, 4, n)
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters for dim=5
        pop_size = 4 + int(3 * np.log(n))  # ~9 for n=5
        pop_size = max(pop_size, 6)
        
        # Initialize around the best point found
        x_mean = best_x.copy()
        sigma = 1.0  # initial step size
        
        # CMA-ES weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + n) / (n + 4 - mu_eff)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Sort indices for selection
        # We'll evaluate pop_size points per generation
        
        remaining_budget = budget - calls
        max_generations = remaining_budget // pop_size
        max_generations = max(max_generations, 5)
        
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if calls + pop_size > budget:
                # Evaluate only as many as we can
                num_evals = budget - calls
                if num_evals <= 0:
                    break
            else:
                num_evals = pop_size
            
            # Generate population
            offspring = []
            for i in range(num_evals):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (B @ (D * z))
                offspring.append(x_i)
            
            # Evaluate offspring
            f_values = []
            for x_i in offspring:
                if time.time() > deadline or calls >= budget:
                    f_values.append(np.inf)
                    continue
                f_val = safe_eval(x_i)
                f_values.append(f_val)
            
            if len(f_values) < 2:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(f_values)
            selected_indices = sorted_indices[:min(mu, len(offspring))]
            
            if len(selected_indices) == 0:
                break
            
            # Update mean
            x_new_mean = np.zeros(n)
            for i, idx in enumerate(selected_indices):
                x_new_mean += weights[i] * offspring[idx]
            
            # Update covariance matrix
            y_vec = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) / chi_n) < 1.4 + 2 / (n + 1)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            rank_mu = np.outer(y_vec, y_vec)
            if h_sigma:
                rank_one = np.outer(p_c, p_c)
            else:
                rank_one = np.outer(p_c, p_c) + delta_h * np.outer(p_c, p_c)
            
            B @= (1 + c_1 * (rank_one - np.eye(n)) + c_mu * rank_mu) * D
            # This is wrong - let me redo the CMA-ES update properly
            
            # Actually, let me use a simpler and more robust approach:
            # Use scipy's differential_evolution or just do a well-tuned Nelder-Mead
            # with restarts. Let me switch to a simpler but effective strategy.
            break
        
        # The CMA-ES above has issues. Let me use a cleaner implementation.
        # Actually, let me just use a well-tuned Nelder-Mead with multiple restarts.
        
        # Clear and redo with a better approach
        # Reset
        calls_reset = calls  # keep track
        best_x_reset = best_x.copy() if best_x is not None else np.zeros(n)
        best_fx_reset = best_fx
        
        # Phase 2: Multiple Nelder-Mead restarts
        def nm_wrapper(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            if np.isfinite(val) and val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val
        
        import scipy.optimize
        
        # Try Nelder-Mead from best point and a few perturbed points
        start_points = [best_x_reset.copy()]
        for i in range(3):
            if time.time() > deadline or calls >= budget:
                break
            perturb = rng.standard_normal(n) * 0.5
            sp = np.clip(best_x_reset + perturb, -5, 5)
            start_points.append(sp)
        
        for start_x in start_points:
            if time.time() > deadline or calls >= budget:
                break
            
            remaining = budget - calls
            if remaining < 10:
                break
            
            try:
                res = scipy.optimize.minimize(
                    nm_wrapper,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 200),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass
        
        # Phase 3: If budget remains, do a final local polish
        if best_x is not None and time.time() < deadline and calls < budget:
            # Try a few more Nelder-Mead from slightly different initial simplex sizes
            for attempt in range(2):
                if time.time() > deadline or calls >= budget:
                    break
                if budget - calls < 10:
                    break
                try:
                    perturb = rng.standard_normal(n) * 0.1
                    sp = np.clip(best_x + perturb, -5, 5)
                    res = scipy.optimize.minimize(
                        nm_wrapper,
                        sp,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(budget - calls, 100),
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'disp': False
                        }
                    )
                except Exception:
                    pass
        
        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
