# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """CMA-ES + L-BFGS-B hybrid optimizer for Bent Cigar.
    CMA-ES adaptively models the anisotropic valley structure via covariance
    adaptation, and L-BFGS-B provides fast local convergence for polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Phase 1: CMA-ES from origin - adaptively learns the anisotropic valley
        # Bent Cigar has extreme anisotropy (condition number ~10^8 for dim=5)
        # CMA-ES covariance adaptation aligns search directions with this valley
        cma_budget = int(budget * 0.45)
        
        def run_cma_es(x0, maxfev):
            nonlocal calls, best_f, best_x
            
            n = len(x0)
            lambda_ = max(4, 4 + int(3 * np.log(n)))
            mu = max(1, lambda_ // 2)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            sigma = 2.0
            x_mean = x0.copy()
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            fev_used = 0
            
            def compute_C_inv_sqrt(A):
                """Compute inverse square root of symmetric positive definite matrix."""
                try:
                    L = np.linalg.cholesky(A)
                    L_inv = np.linalg.inv(L)
                    return L_inv.T @ L_inv
                except np.linalg.LinAlgError:
                    return np.eye(n)
            
            C_inv_sqrt = np.eye(n)
            
            for iteration in range(200):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                if fev_used >= maxfev:
                    break
                
                # Generate population
                pop_x = []
                pop_f = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = C_inv_sqrt @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = wrapper(x_new)
                    pop_x.append(x_new)
                    pop_f.append(f_new)
                    fev_used += 1
                
                # Sort by fitness
                sorted_idx = np.argsort(pop_f)
                
                # Update mean
                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += w[i] * pop_x[sorted_idx[i]]
                
                # Update p_sigma
                if sigma > 1e-15:
                    y_mean = (x_mean_new - x_mean) / sigma
                else:
                    y_mean = np.zeros(n)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)
                
                # Update p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                # Update C
                delta_sigma = (1 + c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1)
                C = (1 + c_1 + c_mu * delta_sigma**2 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    if sigma > 1e-15:
                        z_i = C_inv_sqrt @ (x_mean_new - x_mean) / sigma
                    else:
                        z_i = np.zeros(n)
                    C += c_mu * w[i] * (np.outer(z_i, z_i) - C)
                
                # Ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C += (1e-10 - np.min(eigvals) + 1e-15) * np.eye(n)
                except:
                    pass
                
                C_inv_sqrt = compute_C_inv_sqrt(C)
                x_mean = x_mean_new
            
            return x_mean, best_f
        
        # Run CMA-ES from origin
        if calls < budget - 20 and time.time() < deadline:
            x_cma, f_cma = run_cma_es(np.zeros(dim), cma_budget)
        
        # Phase 2: L-BFGS-B polish from best point
        if calls < budget - 5 and time.time() < deadline:
            lbfgs_budget = min(80, budget - calls - 5)
            if lbfgs_budget >= 10:
                try:
                    res = scipy.optimize.minimize(
                        wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'maxfev': lbfgs_budget}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Phase 3: Additional random restarts with L-BFGS-B if budget remains
        n_restarts = 3
        for i in range(n_restarts):
            if calls >= budget - 5 or time.time() > deadline:
                break
            
            x_start = rng.uniform(-5, 5, dim)
            remaining = budget - calls - 5
            maxfev = min(30, remaining)
            if maxfev < 10:
                break
            
            try:
                res = scipy.optimize.minimize(
                    wrapper,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfev': maxfev}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
