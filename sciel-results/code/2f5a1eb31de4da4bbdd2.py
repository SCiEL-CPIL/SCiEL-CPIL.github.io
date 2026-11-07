"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restarted CMA-ES for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Wrap func to count calls and enforce deadline
        def wrapped_func(x):
            nonlocal total_calls
            if total_calls >= self.budget:
                raise ValueError("Budget exceeded")
            total_calls += 1
            return func(x)
        
        # Simple CMA-ES implementation
        def cmaes_run(x0, budget, seed):
            nonlocal best_x, best_f
            rng = np.random.default_rng(seed)
            d = self.dim
            pop_size = max(4, int(4 + 3 * np.log(d)))
            
            x = x0.copy()
            sigma = 1.0
            C = np.eye(d)
            ps = np.zeros(d)
            pc = np.zeros(d)
            
            # Update rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
            
            # Initial evaluation
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            iters = 0
            while total_calls < self.budget and (time.time() < deadline):
                if iters >= budget:
                    break
                iters += 1
                
                # Generate population
                B = np.linalg.cholesky(C)
                y = rng.standard_normal((pop_size, d))
                z = y @ B.T
                x_pop = np.array([x + sigma * zi for zi in z])
                
                # Evaluate population
                f_pop = np.array([wrapped_func(xi) for xi in x_pop])
                
                # Update best
                idx_min = np.argmin(f_pop)
                if f_pop[idx_min] < best_f:
                    best_f = f_pop[idx_min]
                    best_x = x_pop[idx_min].copy()
                
                # Sort by fitness
                order = np.argsort(f_pop)
                y_sorted = y[order]
                f_sorted = f_pop[order]
                
                # Weighted recombination
                y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Step size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.sqrt(np.trace(C)) * y_w
                h_sig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (iters + 1))) / np.sqrt(mu_eff) < (1 + 2 / (d + 1)) * c_sigma
                
                # Covariance matrix adaptation
                if h_sig:
                    pc = (1 - c_c) * pc + c_c * y_w
                C = (1 + c_1 * (h_sig * (2 - c_c) - c_c) * np.outer(pc, pc) + c_mu * np.sum(weights[:mu, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0) - c_1 * C)
                
                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except:
                    C = np.eye(d)
                
                # Update mean
                x = x + sigma * y_w
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                # Project to box
                x = np.clip(x, -5, 5)
            
            return best_x, best_f
        
        # Strategy: Multiple restarts with diverse starting points
        # Use 3 restarts with different seeds and starting points
        n_restarts = 3
        calls_per_restart = self.budget // n_restarts
        
        for i in range(n_restarts):
            if time.time() >= deadline or total_calls >= self.budget:
                break
            
            # Different starting points for each restart
            if i == 0:
                x0 = np.zeros(self.dim)
            elif i == 1:
                x0 = np.random.uniform(-2, 2, self.dim)
            else:
                x0 = np.random.uniform(-4, 4, self.dim)
            
            cmaes_run(x0, calls_per_restart, seed=i * 12345)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
