"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead for BBOB-style Rosenbrock in 5D."""

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
            # Ensure x is a finite 1D array of size n
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

        # Phase 1: Initial Exploration
        # Use a mix of structured and random points to quickly identify a good basin.
        # Rosenbrock minimum is at (1, ..., 1) in standard form, but shifted/rotated here.
        # We sample around 0 and 1, plus random points.
        
        initial_candidates = [
            np.zeros(n),
            np.ones(n) * 1.0,
            np.ones(n) * 0.5,
            np.ones(n) * -1.0,
            np.ones(n) * -0.5
        ]
        
        # Add random points
        n_random = 15
        rand_points = rng.uniform(-4, 4, (n_random, n))
        initial_candidates.extend(rand_points)
        
        for x in initial_candidates:
            if time.time() > deadline or calls >= budget:
                break
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: CMA-ES Optimization
        # Correct CMA-ES implementation
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
        
        # Covariance matrix inverse square root (B * D)
        B = np.eye(n)
        D = np.ones(n)
        C_inv_sqrt = B * D  # This is B @ diag(D)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        max_generations = (budget - calls) // pop_size
        max_generations = max(max_generations, 5)
        
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if calls + pop_size > budget:
                num_evals = budget - calls
                if num_evals <= 0:
                    break
            else:
                num_evals = pop_size
            
            # Generate population
            offspring = []
            f_values = []
            
            for i in range(num_evals):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (C_inv_sqrt @ z)
                # Clip to box to be safe, though func clips anyway
                x_i = np.clip(x_i, -5, 5)
                offspring.append(x_i)
                
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
            y_vec = (x_new_mean - x_mean) / sigma if sigma > 0 else np.zeros(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) / chi_n) < 1.4 + 2 / (n + 1)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            
            # Rank-one and rank-mu updates
            rank_one = np.outer(p_c, p_c)
            if not h_sigma:
                rank_one += delta_h * np.eye(n)
            
            rank_mu = np.zeros((n, n))
            for i, idx in enumerate(selected_indices):
                y_i = (offspring[idx] - x_mean) / sigma if sigma > 0 else np.zeros(n)
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            # Update C_inv_sqrt (approximate update for B and D)
            # A full eigen-decomposition is expensive, so we use the standard update rule
            # C_new = C + c_1 * (p_c p_c^T + delta_h * I) + c_mu * rank_mu
            # We maintain C_inv_sqrt directly via eigen-decomposition of C
            # For simplicity and speed in 5D, we can just update C and recompute sqrt
            # C = C_inv_sqrt.T @ C_inv_sqrt
            C = C_inv_sqrt.T @ C_inv_sqrt
            C_new = C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.maximum(eigvals, 1e-10)
            C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Recompute C_inv_sqrt
            eigvals_sqrt = np.sqrt(eigvals)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / eigvals_sqrt)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Convergence check
            if sigma < 1e-12:
                break

        # Phase 3: Local Polish with Nelder-Mead
        # If we have budget left, use NM for fine-tuning
        if best_x is not None and time.time() < deadline and calls < budget:
            def nm_wrapper(x):
                nonlocal calls, best_x, best_fx
                if time.time() > deadline:
                    raise ValueError("Time limit exceeded")
                if calls >= budget:
                    raise ValueError("Budget exceeded")
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if np.isfinite(val) and val < best_fx:
                    best_fx = val
                    best_x = x_clipped.copy()
                return val

            try:
                res = scipy.optimize.minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(budget - calls, 100),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            if calls < budget:
                safe_eval(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
