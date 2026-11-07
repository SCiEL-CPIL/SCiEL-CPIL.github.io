# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with strict budget enforcement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin

        # 1. Initial Sampling
        rng = np.random.default_rng(42)
        
        # Start with 10 diverse points to find a good starting region
        n_init = 10
        X_init = rng.uniform(-5, 5, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])
        
        idx_best = np.argmin(F_init)
        mean = X_init[idx_best].copy()
        best_f = F_init[idx_best]
        best_x = X_init[idx_best].copy()
        
        # Estimate initial sigma based on spread of top few
        # Use the distance between the best and the median of the top 3
        top_3_idx = np.argsort(F_init)[:3]
        top_3_pts = X_init[top_3_idx]
        sigma = np.max(np.linalg.norm(top_3_pts - mean, axis=1)) / 2.0
        if sigma < 0.1:
            sigma = 0.5
        if sigma > 2.0:
            sigma = 2.0

        calls_made = n_init

        # 2. CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute C_inv_sqrt for efficiency (update every step)
        # For dim=5, eigh is cheap.

        # 3. Main Loop
        while calls_made + lambda_pop <= budget:
            if time.time() > deadline:
                break

            # Generate population
            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            Z = rng.standard_normal((lambda_pop, dim))
            Y = Z @ C_inv_sqrt
            Pop = mean[None, :] + sigma * Y
            
            # Clip to box
            Pop = np.clip(Pop, -5, 5)

            # Evaluate
            F_pop = np.array([func(x) for x in Pop])
            calls_made += lambda_pop

            # Update best
            idx_best_pop = np.argmin(F_pop)
            if F_pop[idx_best_pop] < best_f:
                best_f = F_pop[idx_best_pop]
                best_x = Pop[idx_best_pop].copy()

            # Sort population
            idx_sorted = np.argsort(F_pop)
            Pop_sorted = Pop[idx_sorted]
            F_sorted = F_pop[idx_sorted]

            # Update mean
            w = weights[:mu]
            new_mean = np.sum(w[:, None] * Pop_sorted[:mu], axis=0)

            # Update path accumulators
            y_mean = C_inv_sqrt @ (new_mean - mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made)))) < (1.4 + 2 / (dim + 1)) * chi
            hs = 1.0 if hsig else 0.0
            
            pc = (1 - c_c) * pc + hs * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma

            # Update covariance
            delta_hs = hs * (1 - c_c) - 1
            C = (1 - c_1) * C + c_1 * (np.outer(pc, pc) + delta_hs * C)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            mean = new_mean

            # Convergence check
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
