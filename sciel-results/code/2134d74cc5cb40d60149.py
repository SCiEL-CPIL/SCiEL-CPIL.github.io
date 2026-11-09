"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase hybrid: coarse global grid search followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Phase 1: Coarse global search using a 3^dim grid
        # In 5D, 3^5 = 243 evaluations. Centers are at -3.33, 0, 3.33
        grid_vals = np.array([-10/3, 0.0, 10/3])
        grid_points = np.array(np.meshgrid(*([grid_vals] * dim))).reshape(-1, dim)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        for p in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            p_clipped = np.clip(p, -5, 5)
            f_val = func(p_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p_clipped.copy()

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        remaining_budget = budget - calls_made
        if remaining_budget < 50:
            return best_x, best_f

        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 0.8
        N = 12  # Population size
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # CMA-ES weights
        mu = N // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        w = np.concatenate([w, np.zeros(N - mu)])
        w_sigma = w[:mu].sum()
        
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (w[0] + w[1] - 1) / ((dim + 2) ** 2 + mu))

        for _ in range(remaining_budget):
            if time.time() > deadline:
                break
            
            # Generate offspring
            Y = np.empty((N, dim))
            for i in range(N):
                B = C_sqrt
                z = rng.standard_normal(dim)
                Y[i] = x_mean + sigma * (B @ z)
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate
            F = np.empty(N)
            for i in range(N):
                F[i] = func(Y[i])
                calls_made += 1
                if calls_made >= budget or time.time() > deadline:
                    break
            
            if calls_made >= budget or time.time() > deadline:
                # Update best if found
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = Y[min_idx].copy()
                break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Weighted mean
            y_w = np.sum(w[:, None] * Y[:mu], axis=0)
            
            # Update covariance matrix
            delta_x = y_w - x_mean
            x_mean = y_w
            
            # Rank-one and rank-mu updates for C
            C += (c_1 + c_mu) * np.outer(delta_x, delta_x)
            for i in range(mu):
                z_i = (Y[i] - x_mean) / sigma
                C += c_mu * w[i] * np.outer(z_i, z_i)
            
            # Ensure symmetry
            C = 0.5 * (C + C.T)
            
            # Cholesky decomposition (with fallback)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or use eigenvalue clipping
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = 0.5 * (C + C.T)
                C_sqrt = np.linalg.cholesky(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(delta_x) / np.sqrt(2 * mu) - 1))
            sigma = max(sigma, 1e-5)
            sigma = min(sigma, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
