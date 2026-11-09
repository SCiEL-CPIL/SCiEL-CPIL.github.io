"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Schaffers F7 in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        total_budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        def track_call(x):
            nonlocal calls_used
            val = func(x)
            calls_used += 1
            return val

        def cma_es_run(init_x, sigma_init, budget):
            nonlocal calls_used, best_x, best_f
            
            # CMA-ES Parameters
            n = dim
            # Standard CMA-ES parameters
            mu = max(1, n // 2)  # Number of selected individuals
            # Weights for the selected individuals
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Step size control parameters
            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (1 - cc)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            # State variables
            xmean = init_x.copy()
            sigma = sigma_init
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Initial evaluation
            fx = track_call(xmean)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
                
            local_calls = 1
            
            # Precompute constants
            sqrt_c1_cmu = np.sqrt(c1 + cmu)
            inv_sqrt_c1_cmu = 1.0 / np.sqrt(c1 + cmu)
            
            for _ in range(budget - 1):
                if time.time() >= deadline:
                    break
                if local_calls >= budget:
                    break
                
                # Generate candidates
                B = np.linalg.cholesky(C)
                D = np.diag(np.sqrt(np.diag(C)))
                inv_D = np.diag(1.0 / np.sqrt(np.diag(C)))
                
                ys = []
                fs = []
                
                for i in range(mu):
                    y = rng.standard_normal(n)
                    x_new = xmean + sigma * B @ y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = track_call(x_new)
                    local_calls += 1
                    ys.append(y)
                    fs.append(f_new)
                    
                    if f_new < best_f:
                        best_f, best_x = f_new, x_new.copy()
                    
                    if local_calls >= budget:
                        break

                if len(fs) < mu:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys = np.array(ys)[idx]
                fs = np.array(fs)[idx]
                
                # Update mean
                y_mean = np.sum(w[:, np.newaxis] * ys, axis=0)
                xmean = xmean + sigma * y_mean
                xmean = np.clip(xmean, -5, 5)
                
                # Update step size
                y_sigma = inv_D @ (B @ y_mean)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                
                # Update covariance
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_mean
                
                # Rank-mu update
                H = (ys.T * w) @ ys
                # Rank-1 update
                H += (1 + c1 * cc / (1 - cc) - c1) * np.outer(pc, pc)
                
                C = (1 + c1 + cmu - c1) * C + c1 * (np.outer(pc, pc) + (1 + cc / (1 - cc) - 1) * np.eye(n)) + cmu * H
                # Simplified update formula for stability
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 + cc/(1-cc) - 1) * np.eye(n)) + cmu * H
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Check for positive definiteness
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If not positive definite, reset C to identity (rare but safe)
                    C = np.eye(n)
                    
                # Ensure sigma doesn't get too small or large
                sigma = np.clip(sigma, 1e-8, 10.0)

        # Strategy: Single robust CMA-ES run
        # CMA-ES is generally superior to (1+1)-ES for rotated landscapes.
        # We use a single run to maximize the budget for the adaptive process.
        # Initialization: Center of the box is often a good start for BBOB.
        x0 = np.zeros(dim)
        # Initial sigma: roughly 1/5 of the box width
        sigma0 = 2.0 
        
        cma_es_run(x0, sigma0, total_budget)
        
        # If we still have budget left (e.g., due to early termination or safety margin),
        # perform a local refinement with a smaller sigma
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            remaining = total_budget - calls_used
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            recent = []
            local_calls = 0
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                if local_calls >= remaining:
                    break
                if len(recent) >= 10:
                    sr = sum(recent) / 10.0
                    if sr > 0.33:
                        sigma *= 1.1
                    elif sr < 0.1:
                        sigma *= 0.8
                    else:
                        sigma *= 0.95
                    recent = recent[-10:]
                sigma = np.clip(sigma, 0.001, 0.5)
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = track_call(y)
                local_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    recent.append(1)
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    recent.append(0)
                if len(recent) > 10:
                    recent = recent[-10:]

        return best_x, best_f
# EVOLVE-BLOCK-END
