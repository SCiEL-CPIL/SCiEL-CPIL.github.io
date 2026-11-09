"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim

        # CMA-ES parameters
        N = 4 + int(np.floor(3 * np.log(dim)))  # population size
        N = max(N, dim + 2)
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        count_evals = 0

        def time_left():
            return time.perf_counter() < deadline

        # Generate initial population
        def sample_population():
            nonlocal count_evals
            zs = rng.standard_normal((N, dim))
            ys = zs @ C_sqrt
            xs = x + sigma * ys
            return xs, zs, ys

        xs, zs, ys = sample_population()
        count_evals += N
        fs = np.array([func(xi) for xi in xs])
        order = np.argsort(fs)
        best_x = xs[order[0]].copy()
        best_f = fs[order[0]]

        # Main loop
        while count_evals < self.budget and time_left():
            # Update based on current population
            x_old = x.copy()
            # Weighted mean of best mu
            y_w = np.sum(weights[:, None] * ys[order[:mu], :], axis=0)
            x += c_sigma * (y_w / sigma)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt.T @ (y_w / sigma))
            hs = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (count_evals // N)))
            if hs < 4.4 / chiN + 0.33:
                c_c_new = c_c
                p_c = (1 - c_c) * p_c + c_c_new * (y_w / sigma)
            else:
                c_c_new = c_c
                p_c = (1 - c_c) * p_c

            # Update C
            delta_hs = 1.0 if hs < 4.4 / chiN + 0.33 else 0.0
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None] * np.outer(ys[order[:mu], :], ys[order[:mu], :]), axis=0)
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Recompute C_sqrt via Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12)))

            # Check convergence
            if sigma < 1e-12 or not np.all(np.isfinite(C)):
                break

            # Sample new population
            try:
                xs, zs, ys = sample_population()
            except Exception:
                break
            count_evals += N

            # Evaluate
            fs = np.array([func(xi) for xi in xs])
            order = np.argsort(fs)
            if fs[order[0]] < best_f:
                best_f = fs[order[0]]
                best_x = xs[order[0]].copy()

            # Check time
            if time.perf_counter() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
