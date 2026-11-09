import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                f_val = float(func(x))
            except Exception:
                return float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def cma_es(x0, budget_limit, sigma0=1.0):
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= budget_limit:
                return best_x, best_f

            # CMA-ES Parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            cc = (4 + dim / (4 + dim)) / ((dim + 4) + (4 + dim / (4 + dim)))
            cs = (dim + 1.3) / (dim + 3 + 2 * dim / (dim + 1))
            cmu = min(0.1, (2 * (dim - 2 + 1. / dim)) / ((dim + 2.) * (dim + 2. + 2. / dim)))
            # Update rate for sigma
            damps = 1 + 2 * max(0, np.sqrt((pop_size - 2) / (dim + 2)) - 1) + cs

            # State variables
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            invsqrtH = 1.0
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
            
            # Initial evaluation of mean
            f_mean = call_f(mean)
            if f_mean is None:
                return best_x, best_f

            # Pre-allocate population storage
            # We will process one generation at a time to manage memory and time strictly
            
            while n_calls < budget_limit and time.time() < deadline - 5:
                # Check time
                if time.time() > deadline - 10:
                    break
                
                # Generate population
                xs = np.zeros((pop_size, dim))
                fvals = np.zeros(pop_size)
                
                # Evaluate population
                for i in range(pop_size):
                    # Generate sample
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (B @ (D * z))
                    x = np.clip(x, -5.0, 5.0)
                    xs[i] = x
                    fvals[i] = call_f(x)
                    if fvals[i] is None:
                        return best_x, best_f
                        
                # Sort by fitness
                indices = np.argsort(fvals)
                xs = xs[indices]
                fvals = fvals[indices]
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = xs[0].copy()

                # Check convergence
                # 1. Sigma too small
                if sigma * D[0] < 1e-12:
                    break
                # 2. Condition number of C too high
                if D[0] / D[-1] > 1e14:
                    break
                # 3. Tolerance on function values
                if np.std(fvals[:pop_size//2]) < 1e-14:
                    break
                # 4. Stagnation check (optional, but CMA-ES handles this via sigma update)
                
                # Update mean
                mu_eff = pop_size // 2
                weights = np.zeros(pop_size)
                # Logarithmic weights
                weights[:mu_eff] = np.log(mu_eff + 1) - np.log(1 + np.arange(mu_eff))
                weights /= weights.sum()
                
                new_mean = np.sum(weights[:, None] * xs, axis=0)
                
                # Update path evolution strategies
                y_mean = (new_mean - mean) / sigma
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (y_mean / np.linalg.norm(pc)) if np.linalg.norm(pc) > 0 else ps
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_calls + 1))) < (1.4 + 2 / (dim + 1)) * chiN)
                
                # Update covariance matrix
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                outer_y = np.outer(y_mean, y_mean)
                
                # Update B and D (eigen-decomposition of C)
                # We maintain C implicitly via B and D. 
                # C = B D^2 B^T
                # Update C directly is expensive. Let's update C matrix explicitly for clarity and stability in this small dim case.
                # Actually, standard CMA-ES updates C = (1 + c_mu * delta_hsig - c_mu) * C + c_mu * sum(wi * yi * yi^T)
                # Let's maintain C explicitly.
                
                # Re-initialize C if not present? No, let's just use C matrix directly.
                # Let's refactor to use C matrix directly for simplicity in this implementation.
                
                # Wait, I started with B/D. Let's stick to C matrix for simplicity of code correctness in a short window.
                # Let's restart the loop logic with C matrix.
                
                # Actually, let's just break and use a simpler, robust CMA-ES variant that uses C matrix.
                # The B/D decomposition is prone to numerical issues if not handled perfectly.
                
                # Let's abort this specific iteration and rely on the fact that for dim=5, we can just use C matrix.
                # I will rewrite the core loop to use C matrix.
                
                break # Exit this specific implementation attempt to use the C-matrix version below

        # --- Robust CMA-ES using C Matrix ---
        
        def robust_cma_es(x0, budget_limit, sigma0=1.0):
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= budget_limit:
                return best_x, best_f

            # Parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            cc = (4 + dim / (4 + dim)) / ((dim + 4) + (4 + dim / (4 + dim)))
            cs = (dim + 1.3) / (dim + 3 + 2 * dim / (dim + 1))
            cmu = min(0.1, (2 * (dim - 2 + 1. / dim)) / ((dim + 2.) * (dim + 2. + 2. / dim)))
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
            
            # State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Weights
            mu_eff = pop_size // 2
            weights = np.zeros(pop_size)
            weights[:mu_eff] = np.log(mu_eff + 1) - np.log(1 + np.arange(mu_eff))
            weights /= weights.sum()
            
            # Initial evaluation
            f_mean = call_f(mean)
            if f_mean is None:
                return best_x, best_f

            gen = 0
            while n_calls < budget_limit and time.time() < deadline - 5:
                if time.time() > deadline - 10:
                    break
                
                xs = np.zeros((pop_size, dim))
                fvals = np.zeros(pop_size)
                
                # Generate and Evaluate
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback if C becomes singular
                        L = np.eye(dim)
                    
                    x = mean + sigma * (L @ z)
                    x = np.clip(x, -5.0, 5.0)
                    xs[i] = x
                    fvals[i] = call_f(x)
                    if fvals[i] is None:
                        return best_x, best_f
                        
                # Sort
                indices = np.argsort(fvals)
                xs = xs[indices]
                fvals = fvals[indices]
                
                # Update Best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = xs[0].copy()

                # Convergence Checks
                if sigma * np.linalg.cholesky(C)[0,0] < 1e-12:
                    break
                if np.linalg.cholesky(C)[0,0] / np.linalg.cholesky(C)[-1,-1] > 1e14:
                    break
                if np.std(fvals[:mu_eff]) < 1e-14:
                    break
                
                # Update Mean
                new_mean = np.sum(weights[:, None] * xs, axis=0)
                
                # Update Paths
                y_mean = (new_mean - mean) / sigma
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (y_mean / np.linalg.norm(pc)) if np.linalg.norm(pc) > 0 else ps
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chiN)
                
                # Update Covariance
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                # C <- (1 + c_mu * delta_hsig - c_mu) * C + c_mu * sum(wi * yi * yi^T)
                C = (1 + cmu * delta_hsig - cmu) * C
                for i in range(mu_eff):
                    yi = (xs[i] - mean) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)
                    
                # Update Sigma (CSA)
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                
                # Update Mean
                mean = new_mean
                
                # Symmetrize C
                C = (C + C.T) / 2
                
                gen += 1
                
            return best_x, best_f

        # Strategy:
        # 1. Global Phase: Run CMA-ES from center (0,0,0,0,0) with large sigma.
        #    Budget: ~200 calls.
        # 2. Local Phase: If budget remains, run CMA-ES from best_x with small sigma.
        
        # Phase 1: Global Exploration
        x_start = np.zeros(dim)
        sigma_global = 2.0
        budget_global = 200
        
        _, _, _ = robust_cma_es(x_start, budget_global, sigma_global)
        
        # Phase 2: Local Refinement
        if n_calls < budget - 10 and time.time() < deadline - 10:
            x_start = best_x.copy()
            sigma_local = 0.1
            budget_local = budget - n_calls - 5
            
            _, _, _ = robust_cma_es(x_start, budget_local, sigma_local)
            
        # Phase 3: Final Polish with (1+1)-ES if budget remains
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.05
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.05
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma_poly *= 0.95
                    sigma_poly = max(sigma_poly, 1e-6)

        return best_x, best_f
