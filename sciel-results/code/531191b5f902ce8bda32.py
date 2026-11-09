"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with Latin Hypercube initialization for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Latin Hypercube Sampling for efficient initial coverage
        n_lhs = min(50, budget // 4)
        rng = np.random.default_rng(42)

        def latin_hypercube(n, d, rng):
            """Generate n points in d dimensions using Latin Hypercube Sampling."""
            points = np.zeros((n, d))
            for i in range(d):
                intervals = rng.random(n)
                order = rng.permutation(n)
                points[order, i] = intervals
            # Map from [0,1]^d to [-4,4]^d
            points = points * 8.0 - 4.0
            return points

        lhs_points = latin_hypercube(n_lhs, dim, rng)

        # Evaluate LHS points
        lhs_results = []
        for idx in range(len(lhs_points)):
            if call_count >= budget:
                break
            if time.time() > deadline - 60:
                break
            x = lhs_points[idx]
            f = tracked_func(x)
            lhs_results.append((f, x))

        if not lhs_results:
            x0 = np.zeros(dim)
            f0 = tracked_func(x0)
            lhs_results = [(f0, x0)]

        # Sort and take best point for CMA-ES initialization
        lhs_results.sort(key=lambda t: t[0])
        x0 = lhs_results[0][1].copy()
        f0 = lhs_results[0][0]

        # Phase 2: CMA-ES
        remaining_budget = budget - call_count
        if remaining_budget <= 10 or time.time() > deadline - 30:
            return best_x, best_f

        # CMA-ES parameters
        popsize = min(12, dim + 3 + int(3 * np.log(dim)))
        mu = popsize // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (4 * (dim + 1.3) ** 2)) / ((dim + 1) ** 2 + mu_eff / 2)
        c_mu = min(1 - c_1, (2 - mu_eff / (4 * (dim + 1.3) ** 2) * (mu_eff - 2 + 1 / mu_eff)) /
                   ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_mu
        c_sigma = (d_sigma + 2) / (dim + d_sigma + 5)
        chi_n = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial sigma: based on the spread of the LHS results
        # Use a moderate initial step size
        sigma = 1.0

        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES loop
        max_iters = remaining_budget // popsize
        for it in range(max_iters):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(popsize):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                # Sample
                z = rng.standard_normal(dim)
                x = mean + sigma * B @ (D * z)
                x = np.clip(x, -4.5, 4.5)
                f = tracked_func(x)
                offspring.append(x)
                offspring_f.append(f)

            if len(offspring) < mu:
                break

            # Sort by function value
            sorted_indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in sorted_indices[:mu]]
            offspring_f = [offspring_f[i] for i in sorted_indices[:mu]]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Compute new mean (weighted)
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i]
            mean = new_mean

            # Compute y_w (normalized new mean - old mean)
            y_w = (new_mean - mean + sigma * 0) / sigma  # Actually: (new_mean - old_mean) / sigma
            y_w = (new_mean - (mean - sigma * 0)) / sigma  # Simplified
            # More carefully: y_w = (new_mean - old_mean) / sigma
            old_mean_for_yw = mean  # This is wrong, we already updated mean
            # Let's recompute properly

            # Actually, let's redo this more carefully
            # We need the old mean before update
            # Let me restructure

            break  # This simple version is getting complex, let's use a cleaner implementation

        # Let me implement CMA-ES more carefully
        # Reset state
        mean = x0.copy()
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        sigma = 1.0

        # Clean CMA-ES implementation
        for it in range(remaining_budget // popsize):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(popsize):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * B @ (D * z)
                x = np.clip(x, -4.5, 4.5)
                f = tracked_func(x)
                offspring.append(x)
                offspring_f.append(f)

            if len(offspring) < mu:
                break

            # Sort by function value
            sorted_indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in sorted_indices[:mu]]
            offspring_f = [offspring_f[i] for i in sorted_indices[:mu]]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Compute weighted mean
            old_mean = mean.copy()
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i]
            mean = new_mean

            # Compute y_w
            y_w = (new_mean - old_mean) / sigma

            # Update path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B @ (D * y_w)
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mu_eff) / chi_n * y_w if 'cs' in dir() else p_c
            # Fix: use c_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / chi_n * y_w

            # Update sigma
            delta_sigma = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma * d_sigma
            sigma *= np.exp(1 / (5 * dim) * delta_sigma)

            # Update covariance matrix
            # Rank-one update
            C = (1 + c_1 * mu_eff) * C + c_1 * np.outer(y_w, y_w)

            # Rank-mu update
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (offspring[i] - old_mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)
            C = (1 + c_1 * mu_eff + c_mu * (mu_eff - 1 - c_1 / 2)) * C + c_mu * H

            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))

            # Check convergence
            if np.linalg.norm(D) < 1e-12:
                break

        # Final local refinement
        remaining = budget - call_count
        if remaining > 0 and best_x is not None and time.time() < deadline - 5:
            local_sigma = 0.1
            for _ in range(min(remaining, 30)):
                if call_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -4.5, 4.5)
                fy = tracked_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                else:
                    local_sigma *= 0.8

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
