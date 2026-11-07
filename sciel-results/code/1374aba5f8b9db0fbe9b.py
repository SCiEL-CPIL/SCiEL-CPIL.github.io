import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Random Initialization
        n_init = 20
        init_points = []
        for _ in range(n_init):
            if n_calls >= budget or time.time() > deadline: break
            x = np.random.uniform(-4.5, 4.5, dim)
            f = timed_call(x)
            init_points.append((f, x.copy()))
        
        if n_calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort by fitness
        init_points.sort(key=lambda x: x[0])
        x_best_init = init_points[0][1].copy()

        # 2. CMA-ES Implementation
        def run_cma_es(x_start, max_calls):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start
            
            n = dim
            # Population size
            lambda_pop = int(4 + 3 * np.log(n))
            
            # Initial step size
            sigma = 0.3
            
            # CMA-ES parameters
            c_sigma = 1.0 / (n + 1)
            d_sigma = 1 + 2 / max(1, n) + c_sigma
            c_c = (4 + n / 4) / (n + 4 + 2 / n)
            c_1 = 2 / ((n + 1.3) ** 2 + n)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) - 1))
            
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Initialize mean and covariance
            mean = x_start.copy()
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            sqrt_C = np.eye(n)
            
            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Cholesky decomposition helper
            def update_chol(C):
                nonlocal C, C_inv_sqrt, sqrt_C
                try:
                    # Regularize if needed
                    L = np.linalg.cholesky(C)
                    sqrt_C = L
                    # For C_inv_sqrt, we need inverse of L^T
                    # C = L L^T, so C^{-1} = L^{-T} L^{-1}
                    # We want M such that M M^T = C^{-1}
                    # Let M = L^{-T}
                    C_inv_sqrt = np.linalg.inv(L.T)
                except np.linalg.LinAlgError:
                    # Regularize C
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                    sqrt_C = L
                    C_inv_sqrt = np.linalg.inv(L.T)

            update_chol(C)
            
            # Local best tracking for this CMA run
            local_best_f = best_f
            local_best_x = mean.copy()
            
            while n_calls < budget and time.time() < deadline:
                # Generate population
                ys = []
                fs = []
                for i in range(lambda_pop):
                    if n_calls >= budget or time.time() > deadline:
                        break
                    # Sample
                    z = np.random.standard_normal(n)
                    y = mean + sigma * (sqrt_C @ z)
                    fy = timed_call(y)
                    ys.append(y)
                    fs.append(fy)
                
                if len(fs) == 0:
                    break
                    
                # Sort
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                # Update local best
                if fs[0] < local_best_f:
                    local_best_f = fs[0]
                    local_best_x = ys[0].copy()
                
                # Update mean
                y_w = np.zeros(n)
                for i in range(min(mu, len(ys))):
                    y_w += weights[i] * ys[i]
                    
                # Update p_sigma
                if np.linalg.norm(C_inv_sqrt @ y_w) > (1.4 + 1/n) * np.sqrt((1 - (1 - c_sigma)**2) * (2 - c_sigma) * mu_eff) * sigma:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ y_w)
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                else:
                    p_sigma = (1 - c_sigma) * p_sigma
                    
                # Update mean
                mean = mean + c_sigma * (y_w - mean)
                
                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * (y_w - mean)
                
                # Update covariance
                # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1 - c_c)^2) * C) + c_mu * sum w_i (y_i - mean)(y_i - mean)^T
                delta_h = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 2/(n+1)) * np.sqrt((1 - (1 - c_c)**2) * (2 - c_c) * mu_eff) else 0.0
                rank_c_update = c_1 * (p_c @ p_c.T + (1 - delta_h) * (1 - (1 - c_c)**2) * C)
                
                rank_mu_update = np.zeros((n, n))
                for i in range(min(mu, len(ys))):
                    diff = ys[i] - mean
                    rank_mu_update += weights[i] * (diff @ diff.T)
                rank_mu_update *= c_mu
                
                C = (1 - c_1 - c_mu) * C + rank_c_update + rank_mu_update
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update Cholesky
                update_chol(C)
                
                # Stagnation check
                if local_best_f == best_f and len(fs) > 0 and fs[0] >= fs[-1]:
                    # If no improvement over the whole population and it's flat, maybe break?
                    # Or just continue to explore.
                    pass
                    
                # If sigma is too small, break
                if sigma < 1e-10:
                    break
                    
            return local_best_x

        # Execution Plan:
        # 1. Run CMA-ES from best init point.
        # 2. If it uses all budget, done.
        # 3. If not, we can try a second run from a different point or polish.
        
        x_cma = run_cma_es(x_best_init, budget - n_calls)
        
        # Final Polish with (1+1)-ES if budget remains
        if n_calls < budget and time.time() < deadline:
            x_polish = x_cma.copy()
            sigma = 0.1
            local_best_f = best_f
            local_best_x = x_polish.copy()
            
            while n_calls < budget and time.time() < deadline:
                w = np.random.standard_normal(dim)
                y = x_polish + sigma * w
                fy = timed_call(y)
                
                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()
                    x_polish = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 1.0)
                
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
