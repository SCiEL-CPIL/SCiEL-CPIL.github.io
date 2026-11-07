"""CMA-ES based optimizer with restarts for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with IPOP-style restarts for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0  # Safety margin

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Initial sampling to find a good starting point for CMA-ES
        # We use a small number of deterministic and random starts to avoid wasting budget
        # on a completely wrong basin before the main search begins.
        starts = [np.zeros(dim)]
        # Add some structured starts
        for s in [1.0, -1.0, 2.0, -2.0]:
            starts.append(np.full(dim, s))
        # Add random starts
        for _ in range(10):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: CMA-ES Optimization
        # We will run CMA-ES with restarts.
        # Parameters:
        # - Population size lambda
        # - Initial sigma
        # - Stagnation threshold
        
        # IPOP-CMA-ES style: Restart with larger sigma and larger population if stagnating.
        # Here we keep population fixed but restart sigma and mean if no improvement.
        
        lambda_ = 10  # Population size for dim=5
        # Standard CMA-ES parameters
        mu = lambda_ // 2  # Number of best points to reweight
        # Weights
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_c = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + 2 * c_c
        # Hessian approximation parameters
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        sigma = 1.0  # Initial step size
        # Mean vector
        x_mean = best_x.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        C_inv_sqrt = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Pre-compute eigenvalues and vectors for C
        # We update C and decompose it each generation (cheap for dim=5)
        
        no_improve_count = 0
        restart_count = 0
        max_restarts = 5
        stagnation_limit = 20  # Generations without improvement before restart

        while n_calls[0] < budget and time.time() < deadline:
            # Check for stagnation
            if no_improve_count > stagnation_limit and restart_count < max_restarts:
                restart_count += 1
                # Restart: Reset to best known point with larger sigma
                # Or a fresh random point if the best is likely local
                if rng.random() < 0.5:
                    x_mean = best_x.copy()
                else:
                    x_mean = rng.uniform(-4.5, 4.5, dim)
                    fx_new = safe_func(x_mean)
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_mean.copy()
                
                sigma = max(sigma * 1.5, 1.0)
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                # Reset budget check for this restart phase? No, global budget applies.

            # Generate population
            X = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                # Sample from N(0, C)
                # x = x_mean + sigma * z, where z = C_inv_sqrt @ N(0, I)
                z = rng.standard_normal(dim)
                y = C_inv_sqrt @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5.0, 5.0)
                X[i] = x_new
                F[i] = safe_func(x_new)
                
                if n_calls[0] >= budget or time.time() > deadline:
                    break

            if n_calls[0] >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            F = F[order]
            X = X[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Update mean
            w = weights
            x_mean_new = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
            
            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt.T @ (x_mean_new - x_mean) / sigma)
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls[0] + 1) / dim)) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            
            # C = (1 + mu_eff * (2 - c_c - c_c/(dim+1)) * c_c - c_c * (h_sigma - 1) * c_c) * C + mu_eff * c_c * (p_c * p_c.T + (1 - h_sigma) * c_c / mu_eff * C) + c_c * mu_eff * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(X[:mu] - x_mean, X[:mu] - x_mean), axis=0) / sigma**2
            # Simplified update:
            # Delta term
            delta = (x_mean_new - x_mean) / sigma
            C = (1 + c_c * (2 - c_c) * mu_eff - c_c * (h_sigma - 1) * c_c) * C + c_c * (np.outer(p_c, p_c) + (1 - h_sigma) * C) + c_c * mu_eff * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(X[:mu] - x_mean, X[:mu] - x_mean) / sigma**2, axis=0)
            
            # Enforce symmetry
            C = (C + C.T) / 2.0
            
            # Decompose C
            # Check for PSD
            try:
                vals, vecs = np.linalg.eigh(C)
                # Clip negative eigenvalues
                vals = np.clip(vals, 1e-10, None)
                C = (vecs * vals) @ vecs.T
                C_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            except np.linalg.LinAlgError:
                # Reset to identity if decomposition fails
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0

            x_mean = x_mean_new

        # Phase 3: Polish (if budget remains)
        if n_calls[0] < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.1, 1e-4)
            polish_window = []
            polish_window_size = 5
            
            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    polish_sigma *= 0.7
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                if len(polish_window) >= polish_window_size:
                    success_rate = sum(polish_window) / polish_window_size
                    if success_rate > 0.4:
                        polish_sigma *= 1.3
                    elif success_rate < 0.15:
                        polish_sigma *= 0.75
                    polish_window = polish_window[-polish_window_size // 2:]

                polish_sigma = max(polish_sigma, 1e-8)
                polish_sigma = min(polish_sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
