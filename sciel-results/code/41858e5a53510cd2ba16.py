"""Rewritten program using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size
        lam = max(4, 4 + int(3 * np.log(dim)))
        # Number of weighted points
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 2)) - 1) / (dim + 3) + c_sigma
        
        # Covariance adaptation
        c_c = (4 - mu_eff + dim) / (dim + 4 - mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers [-2, 2] mostly
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Pre-compute Cholesky for initial sampling
        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)

        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def time_left():
            return (deadline - time.time()) > 5.0

        def budget_left():
            return (budget - call_count) > lam + 5

        # Main CMA-ES Loop
        while time_left() and budget_left():
            # Sample points
            ys = rng.multivariate_normal(np.zeros(dim), C, lam)
            # Alternative: rng.normal(0, 1, (lam, dim)) @ C_chol.T
            # Using multivariate_normal is simpler but might be slower. Let's use Cholesky.
            zs = rng.normal(0, 1, (lam, dim))
            ys = zs @ C_chol.T
            
            xs = x_mean + sigma * ys
            # Clip to bounds
            xs = np.clip(xs, -5.0, 5.0)
            
            # Evaluate
            fs = []
            for x in xs:
                if call_count >= budget:
                    break
                f_val = func(x)
                fs.append(f_val)
                call_count += 1
                
            if not fs:
                break

            fs = np.array(fs)
            # Handle NaNs or Infs
            fs = np.where(np.isfinite(fs), fs, np.inf)
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            ys = ys[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Update mean
            w = weights
            x_new = np.dot(w, xs[:mu])
            
            # Evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_chol.T @ (x_new - x_mean) / sigma)
            
            # h_sigma (hysteresis)
            # chi_D = E[||N(0, I)||]
            chi_D = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2/(dim+1)) * chi_D else 0
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_chol.T @ (x_new - x_mean) / sigma)
            
            # Update C
            # Rank-1 update
            delta_h = (2 - h_sigma) * c_c * (1 - c_c)
            C = (1 + c_1 + c_mu - c_c * (1 - c_c) - c_1 + delta_h) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * np.sum(w * np.outer(ys[:mu], ys[:mu]), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update C_inv
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize if singular
                C = C + 1e-8 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_inv = np.linalg.inv(C)
                    C_chol = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset if completely broken
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_chol = np.eye(dim)
                    p_c = np.zeros(dim)
                    p_sigma = np.zeros(dim)
                    sigma = 1.0

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_D - 1))
            
            # Update mean
            x_mean = x_new

            # Optional: If sigma becomes too small, we might have converged
            if sigma < 1e-8:
                break

        # Final polish if time/budget permits
        if best_x is not None and time_left() and (budget - call_count) > 10:
             # A few Nelder-Mead steps can help fine-tune
             from scipy.optimize import minimize
             try:
                 res = minimize(
                     func,
                     best_x,
                     method='Nelder-Mead',
                     options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True, 'disp': False}
                 )
                 if res.fun < best_f:
                     best_f = res.fun
                     best_x = res.x.copy()
             except Exception:
                 pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
