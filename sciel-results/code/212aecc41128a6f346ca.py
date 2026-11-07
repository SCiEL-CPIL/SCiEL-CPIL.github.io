import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        n = self.dim
        budget = self.budget
        call_count = 0
        best_x = None
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        def time_ok():
            return time.time() < deadline

        def run_cmaes(x0, sigma0, max_calls, seed_offset=0):
            nonlocal call_count, best_x, best_f
            local_rng = np.random.default_rng(42 + seed_offset)
            
            # Parameters
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            x = np.clip(x0, -5, 5).copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Evaluate initial point
            fx = func(x)
            call_count += 1
            update_best(x, fx)
            
            # Track history for stagnation
            hist_f = [fx]
            last_improve = 0
            
            while call_count < max_calls and time_ok():
                # Generate population
                B = local_rng.standard_normal((lam, n))
                
                # Compute C^{1/2}
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                Y = x + sigma * (B @ sqrt_C.T)
                
                # Evaluate
                F = []
                Y_valid = []
                for i in range(lam):
                    if call_count >= max_calls or not time_ok():
                        break
                    y = np.clip(Y[i], -5, 5)
                    fy = func(y)
                    F.append(fy)
                    Y_valid.append(y)
                    call_count += 1
                    update_best(y, fy)
                
                if len(F) < mu:
                    break
                
                F_arr = np.array(F)
                Y_arr = np.array(Y_valid)
                
                # Sort
                idx = np.argsort(F_arr)
                Y_sorted = Y_arr[idx]
                F_sorted = F_arr[idx]
                
                # Update mean
                x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update step size
                y_invC = (Y_sorted[:mu] - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (w[:mu] @ y_invC))
                
                # Update correlation matrix
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                
                # Update covariance
                norm_ps = np.linalg.norm(ps)
                # Approximate h_pc
                gen = call_count // lam
                # Standard check
                if norm_ps / np.sqrt(1 - (1 - c_sigma)**((gen + 1) / d_sigma)) < (1.4 + 2 / (n + 1)) * np.trace(C) / n:
                    h_pc = 1 - np.exp(-2 + 5 / (n + 1))
                else:
                    h_pc = 0.0
                    
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    y_i = y_invC[i]
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                C = (C + C.T) / 2
                
                x = x_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                
                # Clip x to box
                x = np.clip(x, -5, 5)
                
                # Stagnation check
                if F_sorted[0] < hist_f[-1] - 1e-8:
                    last_improve = call_count
                hist_f.append(F_sorted[0])
                
                # If no improvement for 50 evaluations and sigma is small, break
                if call_count - last_improve > 50 and sigma < 1e-3:
                    break

            return best_x, best_f, call_count

        # Strategy:
        # 1. Global phase: 50 random samples to find a good starting point.
        # 2. Local phase: CMA-ES from the best random point.
        # 3. If stagnation or time allows, restart CMA-ES from a perturbed best point.
        
        # Phase 1: Global Random Search
        n_global = min(50, budget // 5)
        for i in range(n_global):
            if not time_ok() or call_count >= budget:
                break
            x_rand = rng.uniform(-5, 5, n)
            f_rand = func(x_rand)
            call_count += 1
            update_best(x_rand, f_rand)
        
        if not time_ok() or call_count >= budget:
            return best_x, best_f
        
        # Phase 2: CMA-ES from best global point
        # Use most of the remaining budget
        alloc1 = budget - call_count - 10  # Keep 10 for potential restart or safety
        if alloc1 < 20:
            alloc1 = budget - call_count
            
        run_cmaes(best_x, 1.0, call_count + alloc1, seed_offset=0)
        
        if not time_ok() or call_count >= budget:
            return best_x, best_f
        
        # Phase 3: Restart if needed
        # If we still have budget, try another CMA-ES run from a slightly perturbed best_x
        remaining = budget - call_count
        if remaining > 20:
            # Perturb best_x
            x_restart = best_x + rng.normal(0, 0.1, n)
            x_restart = np.clip(x_restart, -5, 5)
            run_cmaes(x_restart, 0.5, budget, seed_offset=1)
        
        return best_x, best_f
