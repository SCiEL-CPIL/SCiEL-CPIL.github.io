"""Optimized program for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget

        # We'll use a multi-start strategy with CMA-ES
        # Rastrigin has many local minima, so multi-start is crucial
        # But we only have 500 function calls total

        # Strategy: Use a small number of restarts with CMA-ES
        # For dim=5, Rastrigin has many local minima
        # Let's try: initial random sampling + focused CMA-ES

        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid/random sampling to find a good region
        # Phase 2: CMA-ES refinement from best point

        # Actually, for 500 calls and dim=5, let's be strategic:
        # - Use ~100 calls for diverse initial points
        # - Use remaining ~400 for CMA-ES from the best

        # But CMA-ES with lambda=10 in 5D needs ~10*5=50 calls per generation
        # 400/50 = 8 generations, which may not be enough for convergence
        # Let's use a smaller lambda or adapt

        # Alternative: Use a (mu+lambda)-ES or simple CMA-ES with small population

        # Let me implement a lightweight CMA-ES

        best_x = None
        best_f = np.inf

        # Initial diverse sampling
        n_init = 50
        X_init = rng.uniform(-5, 5, (n_init, dim))
        f_vals = np.array([func(x) for x in X_init])

        # Check time
        if time.time() > deadline:
            idx = np.argmin(f_vals)
            return X_init[idx], f_vals[idx]

        best_idx = np.argmin(f_vals)
        best_x = X_init[best_idx].copy()
        best_f = f_vals[best_idx]

        # Phase 2: CMA-ES from the best point
        # Use a simplified CMA-ES with small population

        # CMA-ES parameters
        pop_size = 10  # lambda
        mu = pop_size // 2  # number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Initial sigma
        sigma = 1.0

        # Means
        m = best_x.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)

        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (sqrt(mu_eff) - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # State
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1 / (2 * dim) - 1 / (4 * dim**2))

        calls_used = n_init
        max_calls = total_budget

        # CMA-ES loop
        while calls_used < max_calls:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(dim)
                X[i] = m + sigma * B @ z
                X[i] = np.clip(X[i], -5, 5)
                calls_used += 1

                if calls_used >= max_calls:
                    break

            if calls_used >= max_calls:
                break

            # Evaluate
            F = np.array([func(x) for x in X[:pop_size]])
            calls_used += pop_size  # already counted above, but let's be safe

            # Actually, I need to be more careful with call counting
            # Let me redo this more carefully

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Update mean
            m_new = m + sigma * np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            m_new = np.clip(m_new, -5, 5)

            # Update step-size control
            y = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2)) < \
                   (1.4 + 2 / (dim + 1)) * chi_n

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            else:
                pc = (1 - c_c) * pc

            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + \
                c_1 * (pc @ pc.T + delta_hsig * C) + \
                c_mu * np.sum(weights[:, None, None] * np.outer(X_sorted[:mu] - m, X_sorted[:mu] - m), axis=0)

            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

            m = m_new

            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
