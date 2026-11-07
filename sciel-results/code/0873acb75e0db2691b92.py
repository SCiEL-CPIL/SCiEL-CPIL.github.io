"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with Nelder-Mead local polishing for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f, call_count
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: CMA-ES for global search ---
        # Initialize mean
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        record(x, fx)

        # CMA-ES parameters
        sigma = 2.0
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        hsig = 1.4 + 2 / (dim + 1)
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES loop
        cma_budget = min(budget - 50, 300)  # Reserve some for local polish
        for gen in range(cma_budget // lam + 1):
            if not time_ok() or call_count >= budget - 50:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                record(y, fy)
                offspring.append((fy, y))

            # Sort
            offspring.sort(key=lambda t: t[0])

            # Update step size
            old_sigma = sigma
            y_mean = (weights @ np.array([offspring[i][1] for i in range(mu)]))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean / (sigma * chi_n)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update covariance
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = offspring[i][1] - x
                rank_mu += weights[i] * np.outer(diff, diff)
            rank_c = (1 - hsig) * np.outer(pc, pc)
            C = (1 + c_1 * (rank_c - C) + c_mu * (rank_mu - C)) * C
            # Regularize
            C = (C + C.T) / 2
            # Ensure positive definite
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback to identity if not PD
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            # Update mean
            x = np.clip(weights @ np.array([offspring[i][1] for i in range(mu)]), -5, 5)
            fx = func(x)
            record(x, fx)

            # Check stagnation
            if np.linalg.norm(y_mean) < 1e-14:
                break

        # --- Phase 2: Nelder-Mead local polish ---
        # Use scipy.optimize.minimize for Nelder-Mead
        from scipy.optimize import minimize

        remaining = budget - call_count
        if remaining > 10 and time_ok():
            # Start NM from best point
            x0 = best_x.copy()
            # Also try a few other promising points if budget allows
            nm_starts = [x0]
            # Add a few random perturbations around best
            for _ in range(min(3, remaining // 20)):
                pert = x0 + 0.1 * rng.standard_normal(dim)
                pert = np.clip(pert, -5, 5)
                nm_starts.append(pert)

            for x_start in nm_starts:
                if not time_ok() or call_count >= budget - 5:
                    break

                # Use a callback to track function calls
                nm_calls = [0]

                def nm_func(x):
                    nonlocal call_count
                    nm_calls[0] += 1
                    x_clip = np.clip(x, -5, 5)
                    f = func(x_clip)
                    record(x_clip, f)
                    return f

                try:
                    res = minimize(nm_func, x_start, method='Nelder-Mead',
                                   options={'maxiter': remaining, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
