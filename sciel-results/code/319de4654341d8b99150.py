"""Optimizer for BBOB f17 (Schaffers F7) with CMA-ES + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with multi-restart and Nelder-Mead/L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = 0
        max_calls = self.budget

        def safe_eval(x):
            nonlocal call_count
            if call_count >= max_calls:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            if len(x_clipped) != self.dim:
                x_clipped = x_clipped[:self.dim]
            f = func(x_clipped)
            call_count += 1
            return f

        # Phase 1: Multi-start CMA-ES
        # Use 3 restarts with different initial points and sigmas
        n_restarts = 3
        calls_per_restart = (self.budget - 20) // n_restarts  # reserve 20 for polish
        
        for r in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Different starting points for diversity
            if r == 0:
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 3.0
            elif r == 1:
                x0 = rng.uniform(-3, 3, self.dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-1, 1, self.dim)
                sigma0 = 1.5
            
            # CMA-ES parameters
            lam = 8  # population size
            mu = lam // 2  # recombination weight count
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            
            x = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            C = np.eye(self.dim)
            C_sqrt_inv = np.eye(self.dim)
            
            # Precompute C^{-1/2} initially
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Track best within this restart
            restart_best_x = x.copy()
            restart_best_f = safe_eval(x)
            if restart_best_f is None:
                break
            if restart_best_f < best_f:
                best_f = restart_best_f
                best_x = x.copy()
            
            # Budget for this restart
            restart_budget = min(calls_per_restart, max_calls - call_count)
            
            for gen in range(restart_budget):
                if time.time() > deadline - 30:
                    break
                if call_count >= max_calls:
                    break
                
                # Generate lambda samples
                z = rng.standard_normal((lam, self.dim))
                # Apply C^{-1/2}
                z = z @ C_inv_sqrt.T
                ys = x[None, :] + sigma * z
                ys = np.clip(ys, -5, 5)
                
                # Evaluate
                fs = np.full(lam, np.inf)
                for i in range(lam):
                    if call_count >= max_calls:
                        break
                    f_i = safe_eval(ys[i])
                    if f_i is not None:
                        fs[i] = f_i
                
                if np.all(np.isinf(fs)):
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                y_sorted = ys[order]
                f_sorted = fs[order]
                
                # Update best
                if f_sorted[0] < restart_best_f:
                    restart_best_f = f_sorted[0]
                    restart_best_x = y_sorted[0].copy()
                    if restart_best_f < best_f:
                        best_f = restart_best_f
                        best_x = y_sorted[0].copy()
                
                # Update mean (recombination)
                y_w = y_sorted[:mu]
                x_new = np.sum(weights[:, None] * y_w, axis=0)
                
                # Update covariance matrix
                y_diff = (y_w - x) / sigma
                delta_x = x_new - x
                C = ((1 + c_c * c_c * (2 - c_c)) * C + 
                     c_c * (np.outer(delta_x / sigma, delta_x / sigma) - C))
                
                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                p_sigma = ((1 - c_sigma) * p_sigma + 
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * 
                          np.dot(weights, y_diff) @ C_inv_sqrt.T)
                p_c = ((1 - c_c) * p_c + 
                       np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_x / sigma)
                
                # Hessian drift correction
                h_sigma = (np.linalg.norm(p_sigma) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) * 
                          (np.sqrt(mu_eff) - 1))
                
                C = C * np.exp(-2 * c_sigma * d_sigma * (np.linalg.norm(p_sigma) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) - 1))
                
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) - 1))
                
                x = x_new
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(self.dim)))
            
            # End of restart r

        # Phase 2: Local refinement from best point
        if best_x is not None and time.time() < deadline - 5:
            # Try Nelder-Mead first
            try:
                res = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
            
            # Try L-BFGS-B for tighter convergence
            if time.time() < deadline - 2:
                try:
                    res2 = minimize(
                        lambda x: func(np.clip(x, -5, 5)),
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5, 5)
                except Exception:
                    pass

        # Final safety check
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
