"""Direct-structure optimizer for BBOB-style f19 (Griewank-Rosenbrock hybrid).

Strategy:
- The problem is a smooth, moderately conditioned 5D function.
- CMA-ES is well-suited for this class.
- We use a single well-tuned CMA-ES run with careful budget management.
- Multiple diverse initializations to handle hidden shifts/rotations.
- Time guard to ensure we finish within the 330s limit.
"""
import numpy as np
import time


class Optimizer:
    """CMA-ES based continuous optimizer for 5D smooth functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 325.0  # 5s safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        used = 0

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Diverse initialization with a small random search
        # Use 10 random points to find a good starting point (save budget for CMA-ES)
        rng = np.random.default_rng(12345)
        init_budget = 10
        candidates = [np.zeros(dim)]
        for _ in range(init_budget - 1):
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        cand_vals = []
        for c in candidates:
            c_clipped = np.clip(c, -5, 5)
            v = func(c_clipped)
            used += 1
            cand_vals.append(v)
            if v < best_f:
                best_f = v
                best_x = c_clipped.copy()
        
        if check_deadline():
            return best_x, best_f

        # Pick the best candidate as starting point for CMA-ES
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]

        # Phase 2: CMA-ES
        # Parameters tuned for 5D smooth optimization
        lam = 14  # population size (larger for better exploration)
        mu = lam // 2  # number of parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # State
        sigma = 2.0  # initial step size, larger for better exploration
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        remaining = total_budget - used
        cma_budget = remaining
        max_iters = cma_budget // lam

        for it in range(max_iters):
            if check_deadline() or used >= total_budget:
                break

            # Sample lambda points from N(x, sigma^2 * C)
            # Use eigendecomposition for stable sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            
            X = []
            F = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                y = eigvecs @ (sqrt_eigvals * z)
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                used += 1
                X.append(xi)
                F.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            X = np.array(X)
            F = np.array(F)

            # Sort by fitness
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]
            
            # Compute Y = (X - x_old) / sigma for the selected parents
            Y = (X_sorted[:mu] - x) / sigma

            # Update mean
            x_new = x + sigma * np.sum(w[:, None] * Y, axis=0)
            x = np.clip(x_new, -5, 5)

            # Update p_sigma
            # Compute C^(1/2) approximation via eigenvalues
            # For simplicity, use the ratio of sigma to update
            p_sigma = (1 - c_sigma) * p_sigma
            # Approximate C^(1/2) * Y / sigma using eigenvalues
            C_sqrt_inv = np.zeros((dim, dim))
            for i in range(dim):
                C_sqrt_inv[:, i] = eigvecs[:, i] * (1.0 / sqrt_eigvals[i])
            p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (Y.T @ w).T
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp((1.0 / (5 * dim + 5)) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (Y.T @ w).T

            # Update covariance matrix
            ys = Y
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(ys[i], ys[i])
            
            factor = (1 - c_c - c_sigma * (1 - c_sigma) + c_sigma ** 2 * (1 - 1.0 / mu_eff))
            C = factor * C + c_sigma * (1 - 1.0 / mu_eff) * H + c_c * (1 - c_c) * (C - np.outer(p_c, p_c))
            
            # Symmetrize and ensure PSD
            C = (C + C.T) / 2
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            if np.min(eigvals_C) < 1e-10:
                C = eigvecs_C @ np.diag(np.clip(eigvals_C, 1e-10, None)) @ eigvecs_C.T

            # Check for stagnation: if sigma is very small, break
            if sigma < 1e-10:
                break

        # Phase 3: Polish with (1+1)-ES if budget remains
        if not check_deadline() and used < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.3, sigma * 0.3)  # smaller initial step for fine-tuning
            
            while used < total_budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05  # gentle increase on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85  # more aggressive decrease on failure
                    if sigma < 1e-10:
                        break

        return best_x, best_f
