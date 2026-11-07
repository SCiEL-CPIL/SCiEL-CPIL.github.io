"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (dim + 2) - 1))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Track cumulative factor for ps norm
        ps_norm_factor = 1.0

        # Evaluate initial point
        x_clipped = np.clip(x, -5, 5)
        try:
            fx = func(x_clipped)
            if not np.isfinite(fx):
                fx = 1e10
        except Exception:
            fx = 1e10
            
        best_x = x_clipped.copy()
        best_fx = fx

        calls_made = 1

        # Population generation
        while calls_made < budget and time.time() < deadline:
            # Generate population
            Y = rng.standard_normal((lambda_, dim))
            
            # Compute C_sqrt once per generation
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset to identity or use eigen
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = np.linalg.cholesky(C)

            P = Y @ C_sqrt.T
            X = x + sigma * P
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.empty(lambda_)
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    break
                try:
                    F[i] = func(X[i])
                    if not np.isfinite(F[i]):
                        F[i] = 1e10
                except Exception:
                    F[i] = 1e10
                calls_made += 1
            
            # If we ran out of budget mid-generation, pad with last known best or large value
            # But since we break out of loop, we should handle partial generation carefully.
            # For simplicity, if we don't have a full generation, break.
            if calls_made < budget + lambda_: # This check is rough, let's just check if we have lambda_ points
                 # Actually, the loop above increments calls_made. 
                 # If we broke early, F has uninitialized values.
                 # We should only proceed if we have a full population.
                 if calls_made < (budget - (budget % lambda_)): 
                     # We don't have a full gen left, break
                     break

            # Ensure all F are finite
            F = np.where(np.isfinite(F), F, 1e10)

            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_fx:
                best_fx = F[idx_best]
                best_x = X[idx_best].copy()

            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Weighted mean
            m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update step size
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (m - x))
            
            # Update ps_norm_factor
            ps_norm_factor *= (1 - c_sigma)
            
            # Check condition for sigma update
            h_sigma = 1.0
            if np.linalg.norm(ps) / np.sqrt(1 - ps_norm_factor) < (1.4 + 2 / (dim + 1)) * chi_n:
                h_sigma = 1.0
                sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / chi_n - 1))
            else:
                h_sigma = 0.0

            # Update mean
            x_old = x.copy()
            x = m

            # Update covariance
            # pc update
            pc = (1 - c_c) * pc + c_c * h_sigma * (m - x_old)

            # C update
            # C = (1 - c_1 - c_mu + c_mu * h_sigma * mu_eff) * C + c_1 * (pc*pc.T - C) + c_mu * sum(w_i (y_i - y_bar)(y_i - y_bar).T)
            # Note: y_i = X_sorted[i] - x_old
            
            Y_diff = X_sorted[:mu] - x_old
            rank_update = np.sum(weights[:, None, None] * np.outer(Y_diff, Y_diff), axis=0)
            
            C = (1 - c_1 - c_mu + c_mu * h_sigma * mu_eff) * C + \
                c_1 * (np.outer(pc, pc) - C) + \
                c_mu * rank_update

            # Ensure positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Final polish: local search around best point
        if time.time() < deadline - 5:
            sigma_local = 0.1
            for _ in range(50):
                if time.time() > deadline:
                    break
                y = best_x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = func(y)
                    if not np.isfinite(fy):
                        fy = 1e10
                except Exception:
                    fy = 1e10
                if fy < best_fx:
                    best_x = y
                    best_fx = fy
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
