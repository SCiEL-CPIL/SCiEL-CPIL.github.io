import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        self.call_count = 0
        dim = self.dim
        budget = self.budget
        
        # Time limit safety margin (e.g., 5 seconds)
        time_limit = 300 - 5 

        def safe_call(x):
            if self.call_count >= budget:
                return None
            self.call_count += 1
            try:
                fx = float(func(x))
                if np.isfinite(fx):
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x.copy()
                    return fx
                else:
                    return 1e100
            except Exception:
                return 1e100

        # 1. CMA-ES Phase
        # Parameters
        lambda_pop = max(4, int(4 + 3 * np.log(dim))) # ~10 for dim=5
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        c_c = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
        c_1 = 2 / (lambda_pop**2 + 1.8)
        c_mu = min(c_1, 2 * (1 - c_1) / (lambda_pop**2 - 2))
        d_sigma = c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (lambda_pop + 1)) - 1) + c_sigma
        
        # Initialization
        rng = np.random.default_rng(0)
        x = np.zeros(dim) # Start at center, usually good
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        hs = np.sqrt((1 + c_c) / (1 - c_c)) + 1
        
        # Precompute Cholesky of C
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(C))
        
        # We need to update B and D when C changes. 
        # For simplicity and speed in low dim, we can recompute Cholesky every step.
        
        best_x_cma = x
        best_f_cma = float('inf')
        
        while self.call_count < budget and (time.time() - self.start_time) < time_limit:
            # Check if we should stop CMA-ES and switch to polish
            # If sigma is very small, CMA-ES is done
            if sigma < 1e-12:
                break
                
            # Generate population
            ys = []
            xs = []
            for _ in range(lambda_pop):
                z = rng.standard_normal(dim)
                # Transform z to y
                y = B @ (D * z)
                x_new = x + sigma * y
                xs.append(x_new)
                ys.append(y)
            
            # Evaluate
            fs = []
            for x_new in xs:
                fx = safe_call(x_new)
                if fx is None:
                    break
                fs.append(fx)
            
            if len(fs) < lambda_pop:
                break # Budget exceeded mid-generation
                
            # Sort
            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            if fs[0] < best_f_cma:
                best_f_cma = fs[0]
                best_x_cma = xs[0].copy()
            
            # Update mean
            x_new_mean = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update step size
            y_mean = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean / np.linalg.norm(C) # Approx norm
            # Actually, norm of C is tricky. Standard uses norm of y_mean / sigma?
            # Let's use standard formula: ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * (B^{-1} D^{-1} y_mean) / sigma?
            # Simpler: ps is in the transformed space?
            # Standard CMA-ES:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B^{-1} * D^{-1} * y_mean)
            # Wait, y = B D z. So z = D^{-1} B^{-1} y.
            # The step is sigma * B D z.
            # The "natural gradient" direction is z.
            # So ps update uses z_mean.
            
            # Let's stick to a simpler robust variant if standard is too complex to get right quickly.
            # A simpler (1+lambda) or (mu+lambda) with fixed sigma adaptation might be safer?
            # No, CMA-ES is key for Rosenbrock.
            
            # Let's use the standard update carefully.
            # y_mean is the weighted sum of y vectors.
            # z_mean = D^{-1} B^{-1} y_mean
            B_inv = np.linalg.inv(B)
            D_inv = 1.0 / D
            z_mean = D_inv * (B_inv @ y_mean)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            
            # Check for step size increase/decrease
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (self.call_count - lambda_pop))) < (1.4 + 2 / (dim + 1)) * hs:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (self.call_count - lambda_pop))) - hs))
                
            # Update mean
            x = x_new_mean
            
            # Update Covariance
            pc = (1 - c_c) * pc + c_c * (x_new_mean - x) / sigma # Wait, x is old mean
            # Correction: pc = (1-c_c)pc + c_c (x_new - x_old)/sigma
            pc = (1 - c_c) * pc + c_c * (y_mean) # y_mean is (x_new - x_old)/sigma
            
            delta_hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (self.call_count - lambda_pop))) < (1.4 + 2 / (dim + 1)) * hs) - 1
            
            rank_mu = np.outer(y_mean, y_mean)
            # Weighted sum of outer products of top mu y's
            # y_i are the sorted y vectors
            for i in range(mu):
                rank_mu += (1 - weights[i]) * 0 # Just placeholder, need to sum w_i * y_i y_i^T
                # Actually: sum_{i=1}^mu w_i (y_i y_i^T)
            # Let's compute rank_mu properly
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i], ys[i])
                
            C = (1 + c_1 * (delta_hs - 1) + c_mu) * C - c_1 * np.outer(pc, pc) + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
            except np.linalg.LinAlgError:
                # If not PD, add small identity
                C += 1e-12 * np.eye(dim)
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
        
        # 2. Polish Phase
        # If we have budget left and time left, run a local optimizer
        if self.call_count < budget and (time.time() - self.start_time) < time_limit - 5:
            start_x = self.best_x
            # Use Nelder-Mead
            # We need to wrap func to count calls
            def wrapped_func(x):
                if self.call_count >= budget:
                    return 1e100
                self.call_count += 1
                try:
                    return float(func(x))
                except:
                    return 1e100
            
            # SciPy's minimize doesn't stop on call count easily without a callback or wrapper that raises.
            # We can use a wrapper that raises an exception if budget exceeded, and catch it.
            class BudgetExceeded(Exception):
                pass
            
            def strict_func(x):
                if self.call_count >= budget:
                    raise BudgetExceeded()
                self.call_count += 1
                try:
                    return float(func(x))
                except Exception as e:
                    if isinstance(e, BudgetExceeded):
                        raise
                    return 1e100
            
            try:
                # maxiter is not call count, but a heuristic.
                # We'll just let it run until it converges or budget is hit.
                res = scipy.optimize.minimize(strict_func, start_x, method='Nelder-Mead', 
                                              options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.success and np.isfinite(res.fun):
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res.x
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f
