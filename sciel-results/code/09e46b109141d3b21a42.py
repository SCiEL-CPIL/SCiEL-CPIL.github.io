"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with proper rank-mu update for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def timed_call(x):
            return func(np.asarray(x, dtype=float))

        best_x = None
        best_f = float('inf')
        calls = 0

        def update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        def time_left():
            return deadline - time.time()

        # --- Phase 0: Initial sampling for early progress ---
        for _ in range(10):
            if calls >= budget or time_left() <= 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            f = timed_call(x)
            calls += 1
            update(x, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float('inf')

        # --- CMA-ES with proper rank-mu update ---
        # Parameters
        lam = 8  # population size
        mu = lam // 2  # number of parents
        # Weights: log-spaced
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c_sigma = 1.0 / (dim + 4)
        c1 = (4 - mu_eff + c_sigma) / ((dim + 3.5) ** 2 + 1)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + 1))
        cs = 1.0 / (dim + 3)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        x_mean = np.array(best_x, dtype=float)
        sigma = 2.0  # initial step size
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def cholesky_update(C):
            """Update B and D from C via Cholesky decomposition."""
            nonlocal B, D
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C)
                B = np.linalg.inv(L)
                D = 1.0 / np.diag(L)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = 1.0 / np.sqrt(eigvals)
                B = eigvecs

        # Initial Cholesky
        cholesky_update(C)

        for step in range(budget):
            if calls >= budget or time_left() <= 5.0:
                break

            # Generate lambda offspring
            Z = rng.standard_normal((lam, dim))
            Y = np.array([x_mean + sigma * (z @ B.T * D) for z in Z])
            Y = np.clip(Y, -5, 5)

            # Evaluate offspring
            F = []
            for y in Y:
                if calls >= budget or time_left() <= 5.0:
                    break
                fy = timed_call(y)
                calls += 1
                F.append(fy)

            if len(F) < lam:
                break

            F = np.array(F)
            order = np.argsort(F)

            # Update best
            best_idx = order[0]
            if F[best_idx] < best_f:
                best_f = F[best_idx]
                best_x = np.array(Y[best_idx], dtype=float)

            # Compute weighted mean of top mu
            Y_w = Y[order[:mu]]
            new_mean = np.sum(weights[:, None] * Y_w, axis=0)

            # Update covariance matrix
            # Rank-one update with pc
            y_vec = (new_mean - x_mean) / sigma
            pc = (1 - cs) * pc + cs * y_vec

            # Hessian of pc
            hsig = (np.sqrt(1 - (1 - cs) ** (2 * (step + 1))) / chi - 1) * cs * damps

            # Rank-mu update
            Y_mu = (Y_w - x_mean) / sigma
            rank_mu = np.sum(weights[:, None] * np.outer(Y_mu, Y_mu), axis=0)

            C = (1 - c_sigma * hsig - c1 - cmu) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * rank_mu

            # Symmetrize and regularize
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            # Update Cholesky
            cholesky_update(C)

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(cs * (2 - cs) * damps) * y_vec
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi < (1.4 + 2 / (dim + 1)) * chi else 0
            sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi - 1))

            # Update mean
            x_mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
