"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin within 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def bounded_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES from x0 with initial step size sigma0, using at most max_calls evaluations."""
            dim = len(x0)
            x = x0.copy()
            sigma = sigma0
            fx = bounded_func(x)
            calls = 1

            # CMA-ES parameters
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            pc = np.zeros(dim)
            hs = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            pc_norm = 0.0

            while calls < max_calls:
                if time.time() > deadline:
                    break

                # Generate lambda samples
                z = rng.standard_normal((lam, dim))
                # Transform using C
                y = np.array([z[i] @ B * D for i in range(lam)])
                x_pop = np.array([x + sigma * y[i] for i in range(lam)])
                x_pop = np.clip(x_pop, -5.0, 5.0)

                # Evaluate
                f_pop = np.array([bounded_func(x_pop[i]) for i in range(lam)])
                calls += lam

                # Sort
                order = np.argsort(f_pop)
                x_pop = x_pop[order]
                f_pop = f_pop[order]

                # Update mean
                x_new = np.sum(weights[:, None] * x_pop[:mu], axis=0)
                x_new = np.clip(x_new, -5.0, 5.0)

                # Update step size
                y_mean = np.sum(weights[:, None] * (x_pop[:mu] - x), axis=0) / sigma
                hs = np.sqrt((1 - (1 - c_sigma) ** 2) / (1 - c_sigma)) * np.linalg.norm(y_mean) / chi_n
                pc = (1 - c_sigma) * pc + c_sigma / hs * (y_mean if hs > 1e-20 else 0)
                pc_norm = np.linalg.norm(pc)
                hsig = (pc_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) / chi_n < 1.4 + 2 / (dim + 1))
                sigma *= np.exp(c_sigma / d_sigma * (pc_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                # Update covariance
                yw = np.array([(x_pop[i] - x) / sigma for i in range(mu)])
                delta_hsig = (1 - hsig) * c_sigma * (2 - c_sigma)
                C = ((1 - (c_1 + c_mu) + c_mu * delta_hsig) * C
                     + c_1 * (np.outer(pc, pc) + (1 - hsig) * delta_hsig * C)
                     + c_mu * (yw.T @ np.diag(weights) @ yw))

                # Symmetrize and recompute eigendecomposition periodically
                C = 0.5 * (C + C.T)
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = np.sqrt(eigvals)

                x = x_new
                fx = f_pop[0]

            return x, fx, calls

        # Multi-start CMA-ES
        # First, do a quick random scan to find a good starting point
        n_scan = min(20, budget // 10)
        scan_x = rng.uniform(-5, 5, (n_scan, dim))
        scan_f = np.array([bounded_func(scan_x[i]) for i in range(n_scan)])
        calls_made += n_scan  # already counted in bounded_func

        best_scan_idx = np.argmin(scan_f)
        best_x = scan_x[best_scan_idx].copy()
        best_f = scan_f[best_scan_idx]

        # Run CMA-ES from the best scan point with a reasonable budget
        remaining = budget - calls_made
        if remaining > 30:
            x_result, f_result, used = cma_es(best_x, 1.5, remaining, rng)
            if f_result < best_f:
                best_x, best_f = x_result, f_result

        # If we have budget left, try another CMA-ES from a different point
        remaining = budget - calls_made
        if remaining > 50 and time.time() < deadline - 10:
            # Try from a random point near the best
            x2 = best_x + 1.0 * rng.standard_normal(dim)
            x2 = np.clip(x2, -5, 5)
            x_result2, f_result2, used2 = cma_es(x2, 1.0, remaining, rng)
            if f_result2 < best_f:
                best_x, best_f = x_result2, f_result2

        return best_x, best_f
# EVOLVE-BLOCK-END
