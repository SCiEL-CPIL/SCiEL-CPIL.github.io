"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        total_calls = 0

        # Restart strategy: Try a few different initializations or restarts
        # For f22, a single long run might get stuck, but CMA-ES is robust.
        # We will implement a simple restart mechanism if progress stalls.
        
        max_restarts = 5
        calls_per_restart = budget // max_restarts
        if calls_per_restart < 50:
            calls_per_restart = 50
            max_restarts = budget // calls_per_restart

        for r in range(max_restarts):
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Initialize CMA-ES
            # Start near center or random? For f22, random is safer, but center is good for first try.
            # Let's use a slightly biased random start or just uniform.
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            
            # If we have a previous best, we could start near it, but for multimodal, random restarts are better.
            # Actually, starting near the global best found so far is often good for exploitation, 
            # but for f22, getting stuck in a local optimum is the main risk. 
            # A restart implies a fresh start.
            
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # Population size
            lambda_pop = 4 + int(3 * np.log(dim))
            if lambda_pop < 2 * dim:
                lambda_pop = 2 * dim
            # For dim=5, lambda = 4 + 3*1.6 = 8.8 -> 8. 2*dim=10. So lambda=10.
            
            # CMA-ES parameters
            alpha_mu = 0.5 * (lambda_pop / 2) / (dim + 2)
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 5)
            cmu = min(1 - c1, 2 * (alpha_mu - 1 + 1 / dim) / ((dim + 2) ** 2 + alpha_mu))
            cs = (dim + 1.7 - 1 / dim) / (dim + 9 + 2 * dim / (lambda_pop / 2))
            ds = 1 + 2 * max(0, np.sqrt((alpha_mu - 1) / (dim + 1)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            calls_in_restart = 0
            
            while total_calls < budget:
                if time.time() - start_time > time_limit:
                    return best_x, best_f
                
                # Sample lambda points
                B = np.linalg.cholesky(C)
                ys = []
                xs = []
                fs = []
                
                for i in range(lambda_pop):
                    if total_calls >= budget:
                        break
                    if time.time() - start_time > time_limit:
                        return best_x, best_f
                        
                    y = rng.standard_normal(dim)
                    xi = x + sigma * (B @ y)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    total_calls += 1
                    calls_in_restart += 1
                    ys.append(y)
                    xs.append(xi)
                    fs.append(fi)
                    
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                if not xs:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                xs_sorted = [xs[i] for i in idx]
                fs_sorted = [fs[i] for i in idx]
                ys_sorted = [ys[i] for i in idx]
                
                # Weights
                weights = np.ones(lambda_pop)
                weights[:int(lambda_pop / 2)] = 1.0
                # Standard weights: w_i = log(lambda/2 + 1) - log(i+1)
                w = np.zeros(lambda_pop)
                for i in range(lambda_pop):
                    w[i] = np.log(lambda_pop / 2 + 1) - np.log(i + 1)
                w = w / w.sum()
                mu_eff = 1.0 / np.sum(w[:int(lambda_pop/2)] ** 2)
                
                # Update mean
                x_old = x.copy()
                x = sum(w[i] * xs_sorted[i] for i in range(lambda_pop))
                
                # Update covariance
                y_w = sum(w[i] * ys_sorted[i] for i in range(lambda_pop))
                h_sigma = (np.sqrt((1 - np.exp(-2 / ds)) * np.exp(-2 / ds))) / chi_n * np.linalg.norm(ps) / np.sqrt(dim)
                
                pc = (1 - cs) * pc + cs * h_sigma * y_w
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff * cs * (2 - cs)) * (y_w - x_old / sigma)
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - h_sigma ** 2) * C)
                C += cmu * sum(w[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(lambda_pop))
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                
                # Stagnation check: If no improvement in last X calls, break to restart
                # This is a heuristic. CMA-ES usually handles this, but for f22, restarting helps.
                # We track best_f outside. If best_f hasn't improved in, say, 50 calls, we might restart.
                # However, simple time/budget loop is safer. Let's just let it run until budget or time.
                # But if sigma becomes very small or very large, it's stuck.
                if sigma < 1e-10 or sigma > 1e10:
                    break
                    
                # If we've used all budget, break outer loop too
                if total_calls >= budget:
                    break
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
