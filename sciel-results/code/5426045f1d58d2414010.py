"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart strategy for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Global best tracking
        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial broad sampling to find promising basins
        # Use a portion of budget for initial exploration
        init_calls = min(50, budget // 10)
        init_points = []
        for i in range(init_calls):
            x = rng.uniform(-5, 5, n)
            f = evaluate(x)
            init_points.append((f, x.copy()))

        # Sort by fitness and pick top diverse points for restarts
        init_points.sort(key=lambda t: t[0])

        # Select up to 5 diverse starting points
        start_points = []
        for f, x in init_points:
            if len(start_points) == 0:
                start_points.append(x)
            else:
                # Check diversity: minimum distance to existing start points
                min_dist = min(np.linalg.norm(x - sp) for sp in start_points)
                if min_dist > 1.0:
                    start_points.append(x)
            if len(start_points) == 5:
                break

        # If not enough diverse points, use the best ones
        while len(start_points) < min(5, len(init_points)):
            start_points.append(init_points[len(start_points)][1].copy())

        # CMA-ES implementation
        def cma_es(x0, max_calls, rng_local):
            nonlocal call_count, best_x, best_f

            n_local = self.dim
            # Population size: lambda = 4 + floor(3 * log(n))
            lam = int(4 + 3 * np.log(n_local))
            if lam < 4:
                lam = 4
            if lam > 50:
                lam = 50

            # CMA-ES parameters
            mu = lam // 2  # number of recombination weights
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Step size
            sigma = 1.0

            # Mean
            mean = x0.copy().astype(float)

            # Covariance matrix
            C = np.eye(n_local)

            # Evolution paths
            p_sigma = np.zeros(n_local)
            p_c = np.zeros(n_local)

            # Parameters
            c_sigma = (mu_eff + 2) / (n_local + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n_local + 1)) + c_sigma
            c_c = (4 + mu_eff / n_local) / (n_local + 4 + 2 * mu_eff / n_local)
            c_sigma = (mu_eff + 2) / (n_local + mu_eff + 5)

            # Hessian update parameters
            c_1 = (2 + 2 * (mu_eff - 1) / (n_local + 1)) / ((n_local + 2) ** 2 + mu_eff)
            c_mu = (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n_local + 2) ** 2 + mu_eff)
            c_mu = min(c_mu, 2 * c_1)

            # Precompute
            chi_n = np.sqrt(n_local) * (1 - 1 / (4 * n_local) + 1 / (21 * n_local ** 2))

            # Inverse square root of C (computed via eigen decomposition)
            def invsqrtm(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.maximum(eigvals, 1e-10)
                eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
                return eigvecs @ (eigvals_inv_sqrt[:, None] * eigvecs.T)

            # Initial B and D
            B = np.eye(n_local)
            D = np.ones(n_local)

            calls_used = 0
            stall_count = 0
            best_local_f = best_f

            for gen in range(200):
                if call_count >= max_calls:
                    break
                if time.time() > deadline - 5:
                    break

                # Generate population
                ys = []
                xs = []
                fs = []
                for i in range(lam):
                    if call_count >= max_calls:
                        break
                    if time.time() > deadline - 5:
                        break
                    y = rng_local.standard_normal(n_local)
                    x = mean + sigma * (B @ (D * y))
                    x = np.clip(x, -5, 5)
                    f = evaluate(x)
                    calls_used += 1
                    ys.append(y)
                    xs.append(x)
                    fs.append(f)

                if calls_used == 0:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = [ys[i] for i in order]
                xs_sorted = [xs[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Update best
                if fs_sorted[0] < best_local_f:
                    best_local_f = fs_sorted[0]
                    stall_count = 0
                else:
                    stall_count += 1

                if stall_count > 15:
                    break

                # Recombination
                y_weighted = np.zeros(n_local)
                for i in range(min(mu, len(ys_sorted))):
                    y_weighted += w[i] * ys_sorted[i]

                # Update evolution path for sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * y_weighted)) / sigma

                # Sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update mean
                mean_old = mean.copy()
                mean_new = np.zeros(n_local)
                for i in range(min(mu, len(xs_sorted))):
                    mean_new += w[i] * xs_sorted[i]
                mean = mean_new

                # Check for convergence
                if np.linalg.norm(mean - mean_old) < 1e-10 * (1 + np.linalg.norm(mean)):
                    if sigma < 1e-10:
                        break

                # Update evolution path for C
                y_mean = np.zeros(n_local)
                for i in range(min(mu, len(ys_sorted))):
                    y_mean += w[i] * ys_sorted[i]

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (D * y_mean)) / sigma

                # Update covariance matrix
                C = (1 + c_1 * (2 - c_c) + c_mu) * C
                C -= c_1 * (2 - c_c) * (np.outer(p_c, p_c) - C)
                for i in range(min(mu, len(ys_sorted))):
                    zi = B.T @ (D * ys_sorted[i])
                    C += c_mu * w[i] * (np.outer(zi, zi) - C)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update B and D via eigen decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-15)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    break

            return best_local_f

        # Run CMA-ES from each starting point
        remaining_budget = budget - call_count
        n_restarts = len(start_points)
        if n_restarts == 0:
            n_restarts = 1
            start_points.append(best_x.copy())

        # Allocate budget to each restart
        budget_per_restart = remaining_budget // n_restarts
        if budget_per_restart < 20:
            budget_per_restart = 20
            n_restarts = max(1, remaining_budget // 20)

        for i in range(n_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            x0 = start_points[i].copy()
            max_calls_for_this = min(budget_per_restart, budget - call_count)
            if max_calls_for_this < 10:
                break

            # Use different seed for each restart
            rng_restart = np.random.default_rng(42 + i * 1000)
            cma_es(x0, max_calls_for_this, rng_restart)

        # Final: use remaining budget for fine-tuning around best point
        if call_count < budget and time.time() < deadline - 5:
            remaining = budget - call_count
            if remaining >= 5:
                # Simple local search around best
                sigma_local = 0.5
                for _ in range(remaining):
                    if time.time() > deadline - 2:
                        break
                    x = best_x + sigma_local * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                    f = evaluate(x)
                    if f < best_f:
                        sigma_local *= 1.2
                    else:
                        sigma_local *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
