"""CMA-ES + multi-method local refinement optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using CMA-ES for global search and multi-method local refinement.
    1. CMA-ES handles hidden rotation/shift via adaptive covariance.
    2. Local refinement: Nelder-Mead → BFGS (finite-diff) → L-BFGS-B.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill limit
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            try:
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return float(fx)
            except Exception:
                calls += 1
                return np.inf

        def finite_diff_gradient(x, f, step=1e-5):
            """Compute gradient via central finite differences."""
            grad = np.zeros(n)
            for i in range(n):
                xp = x.copy()
                xm = x.copy()
                xp[i] += step
                xm[i] -= step
                xp = np.clip(xp, -5.0, 5.0)
                xm = np.clip(xm, -5.0, 5.0)
                grad[i] = (evaluate(xp) - evaluate(xm)) / (2.0 * step)
            return grad

        # --- Phase 1: CMA-ES ---
        lambda_ = 14
        mu = max(1, int(lambda_ / 2))
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 2.8
        
        c_sigma = (1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + mu_eff / (n + 1.0)) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu / n)
        c_mu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0)**2 + mu_eff)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        f0 = evaluate(mean)
        
        cma_budget = int(budget * 0.75)
        
        while calls < cma_budget and time.time() < deadline:
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            
            F = np.array([evaluate(x) for x in X])
            
            if calls >= cma_budget:
                break
                
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            if np.isinf(F_sorted[0]):
                break
                
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            v = (mean_new - mean) / sigma
            try:
                y_p = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y_p = v
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * y_p
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 7.0/(32.0*n**2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / E_norm - 1.0))
            sigma = np.clip(sigma, 1e-12, 5.0)
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                y_c = C_inv_half @ v
            except np.linalg.LinAlgError:
                y_c = v
                
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * y_c
            
            c_mu_adapt = c_mu * (1.0 - c_sigma / d_sigma * (norm_p_sigma**2 - 1.0))
            c_mu_adapt = np.clip(c_mu_adapt, 0.0, 2.0 / mu_eff)
            
            Y_top = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y_top.T).T
            except np.linalg.LinAlgError:
                Y_std = Y_top
                
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            C = (1 - c_mu_adapt) * C + c_mu_adapt * outer_sum
            C = (C + C.T) / 2.0
            
            mean = mean_new
            
            if sigma < 1e-8:
                break

        # --- Phase 2: Multi-method Local Refinement ---
        if not np.isinf(best_f) and calls < budget and time.time() < deadline:
            # Phase 2a: Nelder-Mead for robust local search
            try:
                remaining = budget - calls
                nm_max_iter = max(20, int(remaining * 0.4 / max(1, n)))
                
                res_nm = minimize(
                    lambda x: evaluate(x), 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': nm_max_iter, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = np.clip(res_nm.x, -5.0, 5.0).copy()
            except Exception:
                pass
            
            # Phase 2b: BFGS quasi-Newton with finite-difference gradients
            if not np.isinf(best_f) and calls < budget - 50 and time.time() < deadline:
                try:
                    # Implement BFGS manually to control budget precisely
                    x_bfgs = best_x.copy()
                    f_bfgs = best_f
                    H = np.eye(n)  # Inverse Hessian approximation
                    max_bfgs_iter = 15
                    
                    for it in range(max_bfgs_iter):
                        if calls >= budget - 10 or time.time() > deadline:
                            break
                        
                        # Compute gradient
                        grad = finite_diff_gradient(x_bfgs, func)
                        if not np.all(np.isfinite(grad)):
                            break
                        
                        # Check convergence
                        if np.linalg.norm(grad) < 1e-10:
                            break
                        
                        # Search direction
                        p_dir = -H @ grad
                        
                        # Line search (backtracking)
                        alpha = 1.0
                        c1 = 1e-4
                        f_curr = f_bfgs
                        for ls_iter in range(20):
                            x_trial = np.clip(x_bfgs + alpha * p_dir, -5.0, 5.0)
                            f_trial = evaluate(x_trial)
                            if not np.isfinite(f_trial):
                                alpha *= 0.5
                                continue
                            if f_trial <= f_curr + c1 * alpha * np.dot(grad, p_dir):
                                break
                            alpha *= 0.5
                        
                        if not np.isfinite(f_trial) or f_trial >= f_curr:
                            break
                        
                        # Update BFGS inverse Hessian
                        s = alpha * p_dir
                        grad_new = finite_diff_gradient(x_trial, func)
                        if not np.all(np.isfinite(grad_new)):
                            break
                        
                        y = grad_new - grad
                        sy = np.dot(s, y)
                        if abs(sy) > 1e-12:
                            H = H + (1 + np.dot(y, H @ y) / sy) * np.outer(s, s) / sy - np.outer(s, H @ y) / sy - np.outer(y, H @ s) / sy
                        
                        x_bfgs = x_trial
                        f_bfgs = f_trial
                        grad = grad_new
                        
                        if f_bfgs < best_f:
                            best_f = f_bfgs
                            best_x = x_bfgs.copy()
                except Exception:
                    pass
            
            # Phase 2c: L-BFGS-B for high-precision polish
            if not np.isinf(best_f) and calls < budget and time.time() < deadline:
                try:
                    remaining = budget - calls
                    lbfgs_max_iter = max(5, int(remaining * 0.3 / max(1, n)))
                    
                    res_lbfgs = minimize(
                        lambda x: evaluate(x), 
                        best_x, 
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * n,
                        options={'maxiter': lbfgs_max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res_lbfgs.fun < best_f:
                        best_f = res_lbfgs.fun
                        best_x = np.clip(res_lbfgs.x, -5.0, 5.0).copy()
                except Exception:
                    pass
            
            # Phase 2d: Final tight Nelder-Mead polish
            if not np.isinf(best_f) and calls < budget and time.time() < deadline:
                try:
                    remaining = budget - calls
                    nm2_max_iter = max(10, int(remaining * 0.2 / max(1, n)))
                    
                    res_nm2 = minimize(
                        lambda x: evaluate(x), 
                        best_x, 
                        method='Nelder-Mead', 
                        options={'maxiter': nm2_max_iter, 'xatol': 1e-10, 'fatol': 1e-13}
                    )
                    if res_nm2.fun < best_f:
                        best_f = res_nm2.fun
                        best_x = np.clip(res_nm2.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
