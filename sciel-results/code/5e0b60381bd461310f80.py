import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        # CMA-ES parameters
        # Smaller lambda for faster convergence in low dim, but enough for diversity
        lam = 4 + int(3 * np.log(n)) if n > 1 else 2
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Restart logic
        # If no improvement for 'stagnation_limit' generations, restart from best_x with fresh sigma
        stagnation_limit = 15 
        max_restarts = 10 # Hard cap on restarts to prevent infinite loops if stuck
        
        current_restart = 0
        
        # Initial evaluation at center
        x0 = np.zeros(n)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0

        while calls_made < budget and time.time() < deadline and current_restart < max_restarts:
            # Initialize CMA-ES state for this restart
            # Start from current best, but use a larger sigma to explore neighborhood
            # If this is the first run, sigma=2.0. If restart, sigma=1.0 or smaller?
            # For f13 (sharp ridge), we might be stuck on a ridge. 
            # Restarting from best with a moderate sigma helps.
            sigma = 1.5 if current_restart > 0 else 2.0
            x_mean = best_x.copy()
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            no_improvement_count = 0
            run_best_f = best_f
            
            # Run CMA-ES for this restart
            # We will stop if:
            # 1. Budget exhausted
            # 2. Time limit
            # 3. Stagnation (no improvement for stagnation_limit gens)
            # 4. Sigma too small (converged)
            
            while calls_made + lam <= budget and time.time() < deadline:
                # Generate candidates
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.maximum(vals, 1e-12)
                    L = (vecs * np.sqrt(vals)) @ vecs.T
                
                ys = np.zeros((lam, n))
                for i in range(lam):
                    z = rng.standard_normal(n)
                    ys[i] = x_mean + sigma * L @ z
                    ys[i] = np.clip(ys[i], -5, 5)
                
                # Evaluate
                fs = np.array([func(y) for y in ys])
                calls_made += lam
                
                # Sort indices
                order = np.argsort(fs)
                current_best_in_run = fs[order[0]]
                
                # Update global best
                if current_best_in_run < best_f:
                    best_f = current_best_in_run
                    best_x = ys[order[0]].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Check stagnation
                if no_improvement_count >= stagnation_limit:
                    break
                
                # Update CMA-ES state
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * (ys[order[i]] - x_mean) / sigma
            
                # Update covariance adaptation path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
                # Update step-size control path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
                # Update sigma
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)
                
                # Check if sigma is too small (converged)
                if sigma < 1e-8:
                    break

                # Update mean
                x_mean = x_mean + sigma * y_w
                x_mean = np.clip(x_mean, -5, 5)
                
                # Update Covariance C
                delta_h = (1 - np.exp(-2 * (1 - c_c) * (1))) # Approximation for delta_h in single step context
                h_sig = 1 if norm_ps < (1.4 + 1 / c_sigma) * chi_n else 0
            
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    diff = (ys[order[i]] - x_mean) / sigma
                    rank_mu += w[i] * np.outer(diff, diff)
            
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_sigma * (2 - c_sigma) * (np.outer(ps, ps) - np.trace(C) * np.eye(n))) + c_mu * rank_mu
            
                # Symmetrize
                C = (C + C.T) / 2

            # End of inner loop (stagnation or convergence or budget/time)
            # If we broke due to stagnation or convergence, and we have budget left, restart
            if calls_made < budget and time.time() < deadline:
                current_restart += 1
                # Optional: Small perturbation of best_x for next restart?
                # Usually restarting CMA-ES from best_x with fresh C=I is standard.
                # We already reset C, pc, ps, sigma in the next iteration of the outer loop.
                continue
            else:
                break

        # Use remaining budget for single step (1+1) if available
        if calls_made < budget and time.time() < deadline:
            # Simple (1+1) step using current best
            # Use a small sigma relative to box size
            sigma_11 = 0.5
            while calls_made < budget and time.time() < deadline:
                z = rng.standard_normal(n)
                y = best_x + sigma_11 * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma_11 *= 1.1
                else:
                    sigma_11 *= 0.9
                    sigma_11 = np.clip(sigma_11, 1e-8, 10.0)

        return best_x, best_f
