"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES."""
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 270.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def run_cmaes(x_start, sigma_init, max_calls):
            nonlocal calls_used, best_x, best_f
            
            # CMA-ES parameters
            lam = 10  # Population size
            mu = lam // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # Step size adaptation
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (2 + mueff / dim) / (dim + cc + 2)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cc
            
            # Covariance matrix adaptation
            cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + mueff))
            c1 = (mueff - 2 + 1 / dim) / ((dim + 2) ** 2 - mueff)
            cmu = min(cmu, (1 - 2 * c1) / 2)
            
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # State variables
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            local_calls = 0
            no_improve = 0
            
            while local_calls < max_calls:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # 1. Sample new points
                ys = rng.standard_normal((lam, dim))
                xs = np.array([x_mean + sigma * (ys[i] @ C_sqrt) for i in range(lam)])
                # Clip to box
                xs = np.clip(xs, -5, 5)
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    fs[i] = func(xs[i])
                    calls_used += 1
                    local_calls += 1
                    update_best(xs[i], fs[i])
                
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                
                # 2. Sort
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                ys = ys[order]
                
                # Check for improvement
                if fs[0] < best_f - 1e-12:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 20: # Stagnation check
                        break

                # 3. Update mean
                x_mean_new = np.sum(weights[:, None] * xs[:mu], axis=0)
                
                # 4. Update step size (CSA)
                # Recalculate C_sqrt for ps update (approximate)
                # Ideally, we should maintain C_sqrt via Cholesky decomposition
                # For simplicity and speed, we use the standard update rules
                # Note: This implementation simplifies the C update to avoid full Cholesky
                # if possible, but standard CMA-ES requires it.
                
                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_mean_new - x_mean) / sigma))
                
                # Update sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                
                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma
                
                # Update covariance matrix C
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cc)^2) * C) + cmu * sum(weights[i] * (y_i * y_i.T))
                # Note: The term (1 - (1 - cc)^2) * C is often approximated or handled carefully.
                # Standard formula: C = (1 - c1 - cmu)C + c1(pc pc^T + (1 - (1-cc)^2)C) + cmu * S_w
                # Where S_w = sum(w_i * (y_i * y_i^T))
                
                # We need y_i corresponding to the selected parents
                ys_parents = ys[:mu]
                S_w = np.sum(weights[:, None, None] * (ys_parents[:, :, None] * ys_parents[:, None, :]), axis=0)
                
                C_new = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C) + cmu * S_w
                
                # Ensure C is symmetric
                C_new = (C_new + C_new.T) / 2.0
                
                # Update C_sqrt (Cholesky decomposition)
                try:
                    C_sqrt = np.linalg.cholesky(C_new)
                    # If Cholesky fails, add small jitter
                except np.linalg.LinAlgError:
                    C_new += 1e-9 * np.eye(dim)
                    C_sqrt = np.linalg.cholesky(C_new)
                    
                C = C_new
                x_mean = x_mean_new

            return local_calls

        # Phase 1: CMA-ES
        # Start at 0, with a reasonably large sigma to explore the shifted landscape
        x_start = np.zeros(dim)
        sigma_init = 2.0
        run_cmaes(x_start, sigma_init, 450)
        
        # Phase 2: Fine-tuning (1+1)-ES
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(20):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    update_best(x, fx)
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
