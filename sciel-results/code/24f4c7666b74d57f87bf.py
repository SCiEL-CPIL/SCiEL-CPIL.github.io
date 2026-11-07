import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            calls_made += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random Initialization
        n_init = 10
        init_points = rng.uniform(-5.0, 5.0, size=(n_init, self.dim))
        init_f = np.array([safe_call(p) for p in init_points])
        
        best_idx = np.argmin(init_f)
        mu = init_points[best_idx].copy()
        sigma = 1.0
        
        # Phase 2: CMA-ES
        # Parameters
        lambda_pop = 10
        mu_w = lambda_pop // 2
        
        # Weights
        weights = np.log(mu_w + 0.5) - np.log(np.arange(1, mu_w + 1))
        weights /= np.sum(weights)
        
        # Update rules
        c_sigma = (mu_w + 2) / (self.dim + mu_w + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_w / self.dim) / (self.dim + 4 + 2 * mu_w / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_w)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1 / mu_w) / ((self.dim + 2) ** 2 + mu_w))
        
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Precompute Cholesky factor
        chol = np.linalg.cholesky(C)
        
        while calls_made < self.budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((lambda_pop, self.dim))
            for i in range(lambda_pop):
                z = rng.standard_normal(self.dim)
                pop[i] = mu + sigma * (chol @ z)
                # Clip to bounds? CMA-ES usually handles bounds via reflection or penalty.
                # Here we clip in safe_call, but for sampling, clipping distorts distribution.
                # Better to just clip the input to func, but keep internal state unclipped?
                # If we clip, the effective search space is bounded.
                # Let's clip the candidates before calling func, but update mean based on unclipped?
                # No, if we clip, the "best" might be on the boundary.
                # Standard practice: Clip candidates.
                pop[i] = np.clip(pop[i], -5.0, 5.0)
            
            # Evaluate
            pop_f = np.array([safe_call(x) for x in pop])
            
            # Sort
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Update mean
            new_mu = np.sum(weights[:, np.newaxis] * pop[:mu_w], axis=0)
            
            # Update step size
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_w) / sigma * (C^-1/2 @ (new_mu - mu))
            # C^-1/2 is inverse chol transpose
            inv_chol = np.linalg.inv(chol)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * (inv_chol @ (new_mu - mu)) / sigma
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / self.dim) * np.sqrt(np.trace(C)) / self.dim else 0
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(np.trace(C)) / self.dim - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + h_sigma * c_c * (new_mu - mu) / sigma
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            rank_mu = np.sum(weights[:, np.newaxis] * (pop[:mu_w] - mu)[:, np.newaxis] * (pop[:mu_w] - mu), axis=0) / sigma**2
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(self.dim)
                chol = np.linalg.cholesky(C)
            
            mu = new_mu
            
            # Convergence check
            if sigma < 1e-8 or np.all(np.abs(pop_f[:1] - pop_f) < 1e-10):
                break

        # Phase 3: Polishing with Nelder-Mead
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            # Use remaining budget for Nelder-Mead
            # Nelder-Mead doesn't have a strict "maxiter" in scipy that maps to function calls easily,
            # but we can wrap it.
            # However, scipy's minimize might make many calls.
            # Let's just do a few manual steps or use scipy with a tight limit.
            # Actually, CMA-ES should be close enough.
            # Let's use a few Nelder-Mead iterations.
            x0 = best_x
            # We need to limit function calls.
            # scipy.optimize.minimize doesn't support max_func_calls directly in all versions.
            # We can use a counter.
            nm_calls = 0
            nm_limit = self.budget - calls_made
            
            def nm_func(x):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                return safe_call(x)
            
            try:
                res = minimize(nm_func, x0, method='Nelder-Mead', options={'maxiter': nm_limit, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
