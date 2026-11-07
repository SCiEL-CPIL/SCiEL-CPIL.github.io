"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for BBOB f3 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def remaining_time():
            return deadline - time.time()

        # Use multiple restarts with CMA-ES-like updates
        # For Rastrigin, global search is important initially, then local refinement

        # Phase 1: Global search with random restarts to find a good basin
        # Phase 2: CMA-ES local optimization from the best point

        # First, do some random sampling to find a good starting point
        rng = np.random.default_rng(42)

        # Random initialization - sample several points
        n_init = min(50, budget // 10)
        init_xs = rng.uniform(-5, 5, (n_init, dim))
        init_fs = np.array([func(x) for x in init_xs])
        calls_made += n_init

        best_idx = np.argmin(init_fs)
        x0 = init_xs[best_idx].copy()
        f0 = init_fs[best_idx]

        if best_x is None or f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # If we ran out of budget during init, return
        if calls_made >= budget or remaining_time() < 5:
            return best_x, best_f

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma - for Rastrigin, we want to explore
        sigma = 1.0

        # CMA-ES state
        mean = x0.copy()
        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # CMA-ES constants
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        dampc = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Cholesky factor of C
        C_sqrt = np.linalg.cholesky(C)

        # Track best
        cur_best_x = best_x.copy()
        cur_best_f = best_f

        max_iter = 0
        while calls_made < budget and remaining_time() > 10:
            max_iter += 1
            if max_iter > 10000:  # safety
                break

            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt @ z)
                ys[i] = np.clip(y, -5, 5)

            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls_made += lam

            if calls_made >= budget or remaining_time() < 5:
                # Update best
                for i in range(lam):
                    if fs[i] < cur_best_f:
                        cur_best_f = fs[i]
                        cur_best_x = ys[i].copy()
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update best
            if fs_sorted[0] < cur_best_f:
                cur_best_f = fs_sorted[0]
                cur_best_x = ys_sorted[0].copy()

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * ys_sorted[:mu], axis=0)

            # Update evolution paths
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt.T @ ((mean - mean_old) / sigma))

            # pc update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (max_iter + 1)))) < \
                   (1 + 2 / (dim + 1)) * np.sqrt(dim)
            if hsig:
                pc = (1 - cs) * pc + cs * (mean - mean_old) / sigma
            else:
                pc = (1 - cs) * pc

            # Update covariance
            delta_hsig = 1.0 - hsig
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu * np.sum(weights[i] * np.outer(ys_sorted[i] - mean_old, ys_sorted[i] - mean_old)
                              for i in range(mu))

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-10 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)

            # Restart condition: if sigma is too small or no progress
            if sigma < 1e-6:
                # Try a small random perturbation
                if remaining_time() > 15 and calls_made < budget - lam:
                    mean = cur_best_x + rng.uniform(-0.5, 0.5, dim)
                    mean = np.clip(mean, -5, 5)
                    sigma = 0.5
                    C = np.eye(dim)
                    C_sqrt = np.linalg.cholesky(C)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                else:
                    break

        # Final best
        if cur_best_f < best_f:
            best_f = cur_best_f
            best_x = cur_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
