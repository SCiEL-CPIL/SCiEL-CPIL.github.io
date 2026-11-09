import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        
        class _BudgetExceeded(Exception):
            pass
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            f_val = func(np.array(x, dtype=np.float64))
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: CMA-ES with adaptive strategy
        def cma_es(x0, sigma0, max_iter):
            nonlocal best_x, best_f, eval_count
            
            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(n))
            lambda_ = pop_size
            mu = lambda_ // 2  # number of parents
            c_sigma = (mu_eff + 2) / ((n + 2)**1.5 + mu_eff)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            mu_eff = 1 / sum((i + 1)**-2 for i in range(mu))
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)
            
            # Precompute weights
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            
            iter_count = 0
            while iter_count < max_iter:
                if eval_count >= budget or time.time() > deadline:
                    return
                
                # Generate population
                ys = []
                xs = []
                fs = []
                
                for i in range(lambda_):
                    if eval_count >= budget or time.time() > deadline:
                        return
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    x = x_mean + sigma * y
                    x = np.clip(x, -5, 5)
                    f_val = safe_func(x)
                    ys.append(y)
                    xs.append(x)
                    fs.append(f_val)
                
                # Sort by fitness
                idx = np.argsort(fs)
                ys = [ys[i] for i in idx]
                xs = [xs[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                # Update mean
                x_mean_new = sum(weights[i] * xs[i] for i in range(mu))
                
                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt_inv @ (x_mean_new - x_mean)
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (iter_count + 1)))
                pc = (1 - c_c) * pc + hs < (1.4 + 2 / (n + 1)) * np.sqrt(c_c * (2 - c_c)) * chi_n * (np.sqrt(c_c) * (2 - c_c) / (1 - c_c)) * (x_mean_new - x_mean) / sigma
                
                # Update covariance matrix
                delta_h = (1 - c_c) - c_c
                rank_one = np.outer(pc, pc)
                rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Ensure positive definiteness and recompute sqrt
                C = 0.5 * (C + C.T)
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                    C_sqrt_inv = np.linalg.inv(C_sqrt)
                except:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    C_sqrt_inv = np.eye(n)
                
                x_mean = x_mean_new
                iter_count += 1
        
        # Phase 2: Nelder-Mead local refinement
        def nelder_mead(x0, max_iter=50):
            nonlocal best_x, best_f, eval_count
            
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = safe_func(x0)
            simplex_f = [f0]
            
            for i in range(n):
                if eval_count >= budget or time.time() > deadline:
                    return
                v = x0.copy()
                v[i] += 0.25
                v = np.clip(v, -5, 5)
                simplex.append(v)
                simplex_f.append(safe_func(v))
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            for iteration in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    return
                
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                centroid = np.mean(simplex[1:], axis=0)
                spread = np.max([np.linalg.norm(v - centroid) for v in simplex])
                if spread < 1e-10:
                    break
                
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = safe_func(x_reflect)
                
                if eval_count >= budget or time.time() > deadline:
                    return
                
                if f_reflect < simplex_f[0]:
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = safe_func(x_expand)
                    if eval_count >= budget or time.time() > deadline:
                        return
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    if f_reflect < simplex_f[-1]:
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = safe_func(x_contract)
                    if eval_count >= budget or time.time() > deadline:
                        return
                    if f_contract <= min(f_reflect, simplex_f[-1]):
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_func(simplex[i])
                            if eval_count >= budget or time.time() > deadline:
                                return
                
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
        
        # Initial evaluation at origin
        safe_func(np.zeros(n))
        
        # Run CMA-ES from origin
        cma_es(np.zeros(n), 2.0, max_iter=20)
        
        # Local refinement with Nelder-Mead
        if eval_count < budget and time.time() < deadline:
            nelder_mead(best_x.copy(), max_iter=50)
        
        # Multi-start: try shifted point (Lunacek's second basin)
        if eval_count < budget and time.time() < deadline:
            # The second basin is typically at a shifted location
            # For Lunacek, this is often around the shifted optimum
            x_shifted = np.zeros(n)
            x_shifted[0] = 2.5  # Approximate shift for second basin
            x_shifted = np.clip(x_shifted, -5, 5)
            
            f_shifted = safe_func(x_shifted)
            
            # Run CMA-ES from shifted point
            cma_es(x_shifted, 1.5, max_iter=15)
            
            # Local refinement
            if eval_count < budget and time.time() < deadline:
                nelder_mead(best_x.copy(), max_iter=30)
        
        # Final polish: fine-grained search around best point
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            polish_steps = min(remaining, 50)
            for i in range(polish_steps):
                if time.time() > deadline or eval_count >= budget:
                    break
                scale = 0.1 * (1 - i / polish_steps) + 0.01
                x = best_x + rng.normal(0, scale, n)
                x = np.clip(x, -5, 5)
                safe_func(x)
        
        return best_x, best_f
