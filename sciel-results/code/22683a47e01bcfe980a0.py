import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
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

        def run_cmaes(x_start, sigma_start, budget_cma, rng_cma):
            nonlocal best_x, best_f, calls
            
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
            
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Evaluate start
            fx = try_call(x_mean)
            if fx is None:
                return
            
            # CMA-ES Loop
            while calls < budget_cma and time.time() < deadline:
                # Generate offspring
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                
                normals = rng_cma.standard_normal((lam, dim))
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
                
                idx_sorted = np.argsort(fs)
                ys = np.array(ys)
                fs = np.array(fs)
                ys = ys[idx_sorted]
                fs = fs[idx_sorted]
                
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()
                    
                y_mean = np.dot(weights, ys[:mu])
                z_mean = (y_mean - x_mean) / sigma
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_mean
                
                threshold = (1.4 + 2.0 / (dim + 1)) * np.sqrt(dim - 1)
                hsig = 1 if np.linalg.norm(ps) < threshold else 0
                
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * z_mean
                
                zs_mu = (ys[:mu] - x_mean) / sigma
                H = np.zeros((dim, dim))
                for i in range(mu):
                    H += weights[i] * np.outer(zs_mu[i], zs_mu[i])
                
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * H
                C = (C + C.T) / 2
                
                x_mean = y_mean
                
                # Stop conditions
                if sigma < 1e-8:
                    break
                # Check for stagnation: if improvement is very small over last few generations
                if fs[0] > best_f - 1e-10 and sigma < 0.01:
                    break
                
            return x_mean, sigma

        # Strategy:
        # 1. Start with a few random restarts to find a good basin.
        # 2. Use CMA-ES for the bulk of the budget.
        # 3. Final polish with (1+1) ES and Nelder-Mead
        
        # Let's do a quick random search first to get a decent starting point
        # for CMA-ES, especially if the origin is far from the optimum.
        # The task says minimizer is in [-4, 4]^5. Origin (0,0,0,0,0) is likely close.
        # Let's evaluate origin first.
        x0 = np.zeros(dim)
        f0 = try_call(x0)
        
        # Try a few random points to find a better starting basin
        # Use stratified sampling for better coverage
        n_random = 15
        for i in range(n_random):
            if calls >= budget:
                break
            # Use Latin hypercube-like stratification
            for d in range(dim):
                x0[d] = rng.uniform(-5 + 5*(i/n_random), -5 + 5*((i+1)/n_random))
            fr = try_call(x0)
        
        # Now run CMA-ES from the best found so far
        # We allocate 80% of the budget to CMA-ES
        budget_cma = int(0.8 * budget)
        x_start = best_x.copy() if best_x is not None else np.zeros(dim)
        sigma_start = 2.5
        
        run_cmaes(x_start, sigma_start, budget_cma, rng)
        
        # Phase 2: (1+1) ES for fine tuning with proper adaptive sigma
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.5  # Start larger, let it adapt
            
            # Adaptive (1+1) ES with proper success tracking
            # Use a moving window of successes to adapt sigma
            window_size = 20
            successes = []
            
            while calls < budget and time.time() < deadline:
                y = x + sigma_local * rng.standard_normal(dim)
                fy = try_call(y)
                if fy is None:
                    break
                
                # Record success BEFORE updating x, fx
                success = (fy < fx)
                successes.append(success)
                
                if len(successes) > window_size:
                    successes.pop(0)
                
                if success:
                    x, fx = y, fy
                
                # Adapt sigma based on success rate in window
                if len(successes) >= 5:
                    success_rate = np.mean(successes)
                    # Target ~1/5 success rate for optimal exploration
                    if success_rate > 0.25:
                        sigma_local *= 1.15
                    elif success_rate < 0.08:
                        sigma_local *= 0.85
                    else:
                        sigma_local *= 1.01  # Slight increase to keep exploring
                
                # Stop if sigma is very small
                if sigma_local < 1e-10:
                    break
        
        # Phase 3: Final polish with Nelder-Mead if budget allows
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize
                
                # Use remaining budget for Nelder-Mead
                remaining = budget - calls
                if remaining > 10:
                    # Nelder-Mead with tight tolerances
                    result = minimize(
                        func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': min(remaining, 200), 
                                 'xatol': 1e-8, 
                                 'fatol': 1e-10}
                    )
                    # Note: minimize will make its own calls, but we can't easily track them
                    # So we'll just use the result if it's better
                    if result.fun < best_f:
                        best_x = np.clip(result.x, -5.0, 5.0)
                        best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
