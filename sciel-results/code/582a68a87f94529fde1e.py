import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global search and shape adaptation
        # For dim=5, lambda=10 is a good balance.
        lam = 10
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (mu_eff + 2) / (lam + mu_eff + 5)
        cc = (4 + mu_eff / lam) / (lam + 4 + 2 * mu_eff / lam)
        c1 = 2 / (lam**2 + 18)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / (lam**2 + 18))
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lam + 1)) - 1) + cs
        
        # Initialize CMA-ES
        # Start at origin (or random if origin is bad? Origin is usually decent for BBOB f19)
        x_mean = np.zeros(dim)
        sigma = 2.5 # Initial step size, covering ~1/4 of the box width
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Evaluate initial mean
        fx = try_call(x_mean)
        if fx is None:
            return best_x, best_f
        if best_x is None:
            best_x = x_mean.copy()
            best_f = fx

        # CMA-ES Loop
        while calls < budget and time.time() < deadline:
            # Check if we should stop CMA-ES and switch to local refinement
            # If sigma is very small or we've made few improvements, switch
            # But let's run CMA-ES for a significant portion, say 70% of budget or until sigma < 1e-4
            # For dim=5, 500 evals / 10 = 50 generations.
            
            # Generate offspring
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            
            normals = rng.standard_normal((lam, dim))
            scaled_normals = normals * np.sqrt(eigvals)
            zs = (eigvecs @ scaled_normals.T).T
            
            ys = []
            fs = []
            
            for i in range(lam):
                y = x_mean + sigma * zs[i]
                fy = try_call(y)
                if fy is None:
                    break
                ys.append(y)
                fs.append(fy)
                
            if len(ys) < mu:
                break
            
            # Sort by fitness
            idx_sorted = np.argsort(fs)
            ys = np.array(ys)
            fs = np.array(fs)
            ys = ys[idx_sorted]
            fs = fs[idx_sorted]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Mean update
            y_mean = np.dot(weights, ys[:mu])
            
            # Step size control
            z_mean = (y_mean - x_mean) / sigma
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_mean
            
            # hsig calculation (simplified for stability)
            # Threshold: (1.4 + 2/(dim+1)) * sqrt(dim+1-2)
            threshold = (1.4 + 2.0 / (dim + 1)) * np.sqrt(dim - 1)
            # Correction term for finite horizon is complex, omitting for simplicity/stability
            # norm(ps) / sqrt(1 - (1-cs)^(2*(budget-evals)))
            # For now, just use norm(ps)
            hsig = 1 if np.linalg.norm(ps) < threshold else 0
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Covariance matrix update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * z_mean
            
            # Rank-1 and Rank-mu updates
            zs_mu = (ys[:mu] - x_mean) / sigma
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += weights[i] * np.outer(zs_mu[i], zs_mu[i])
            
            # Standard CMA-ES update
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + (1 - delta_sig) * (||ps||^2/delta_sig^2 - 1) * I) + cmu * H
            # delta_sig term is often omitted or approximated in simple implementations.
            # Let's use the simplified update which is robust:
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * H
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = y_mean
            
            # Early stopping condition for CMA-ES
            if sigma < 1e-5:
                break
                
            # If we have used 70% of budget, stop CMA-ES
            if calls > 0.7 * budget:
                break

        # Phase 2: Local refinement using (1+1)-ES from the best point found
        # This is crucial for the last few orders of magnitude of precision.
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.5 # Start with a moderate step size
            no_improve = 0
            max_no_improve = 50
            
            while calls < budget and time.time() < deadline:
                y = x + sigma_local * rng.standard_normal(dim)
                fy = try_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.1 # Slow increase
                    no_improve = 0
                else:
                    sigma_local *= 0.9 # Slow decrease
                    no_improve += 1
                    
                if no_improve >= max_no_improve:
                    # If stuck, try a larger jump or restart from best
                    if rng.random() < 0.5:
                        # Perturb current best
                        y = best_x + rng.normal(0, 1.0, dim)
                        fy = try_call(y)
                        if fy is None: break
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                            x = y
                            fx = fy
                            sigma_local = 0.5
                            no_improve = 0
                        else:
                            # Keep shrinking
                            sigma_local *= 0.8
                    else:
                        # Just keep shrinking, maybe we're close
                        sigma_local *= 0.8
                        no_improve = 0 # Reset counter to allow more shrinking attempts

        return best_x, best_f
