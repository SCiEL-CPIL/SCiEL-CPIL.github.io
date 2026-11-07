"""Multi-start CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            return func(x)

        # Multi-start CMA-ES
        n_starts = 8
        calls_per_start = max(1, (budget - 1) // n_starts)

        for s in range(n_starts):
            if not time_ok() or calls_made >= budget:
                break

            # Initialize CMA-ES
            if s == 0:
                x0 = np.zeros(dim)
                sigma = 2.0
            else:
                # Random restart, biased towards center
                x0 = rng.uniform(-3, 3, dim)
                sigma = 1.5 + 0.5 * s

            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(dim + 1))  # ~11 for dim=5
            pop_size = min(pop_size, max(4, calls_per_start // 2))
            pop_size = max(4, pop_size)

            # Initial population
            x = x0.copy()
            C = np.eye(dim)
            sigma_current = sigma
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            cc = (1 + 1 / (2 * dim + np.log(dim))) / (pop_size + 2)
            cs_ = (1 + 2 / (dim + 1)) / (pop_size + 2)
            c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            cmu = min(0.3, (2 * (dim - 1.5 + 1 / dim)) / ((dim + 2) ** 2 + pop_size))
            cmu = max(0.1, cmu)
            damps = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1) - 1) - 1) + cs_
            dsigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1) - 1)) + cs_

            # Initial points
            points = []
            fvals = []
            for i in range(pop_size):
                if not time_ok() or calls_made >= budget:
                    break
                y = x + sigma_current * (np.linalg.cholesky(C) @ rng.standard_normal(dim))
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                points.append(y)
                fvals.append(fy)

            if len(points) == 0:
                continue

            # Sort by fitness
            idx = np.argsort(fvals)
            points = [points[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = points[0].copy()

            # Main CMA-ES loop
            while time_ok() and calls_made < budget:
                # Compute mean
                mu = pop_size // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1 / np.sum(weights ** 2)
                xmean = np.sum(weights[:, None] * np.array(points[:mu]), axis=0)

                # Update evolution paths
                diff = (xmean - x) / sigma_current
                cs = (1 - cs_) * cs + cs_ * diff
                pc = (1 - cc) * pc + cc / np.sqrt(damps) * (np.sqrt(mueff) * diff - cs)

                # Check sigma adaptation
                if np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (calls_made - budget + 1))) > (1.4 + 1 / dim) * np.sqrt(dim):
                    sigma_current *= np.exp(cs_ / dsigma * (np.linalg.norm(cs) / np.sqrt(1 - (1 - cs_) ** (calls_made - budget + 1)) - np.sqrt(dim)))
                    sigma_current = max(1e-10, min(sigma_current, 10.0))

                # Update covariance matrix
                delta_sigma = np.exp(cs_ / dsigma * (np.linalg.norm(cs) / np.sqrt(1 - (1 - cs_) ** (calls_made - budget + 1)) - np.sqrt(dim))) - 1
                rank_mu = np.outer(xmean - x, xmean - x)
                C = (1 - c1 - cmu + (1 - c1 - cmu) * delta_sigma + c1 * (1 + (2 - cc / cc) * delta_sigma)) * C \
                    + c1 * (np.outer(pc, pc) + (cc / (1 - cc)) * C) \
                    + cmu / mueff * np.sum(weights[:mu, None, None] * np.outer(np.array(points[:mu]) - xmean, np.array(points[:mu]) - xmean), axis=0)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim)

                x = xmean

                # Generate new population
                new_points = []
                new_fvals = []
                for i in range(pop_size):
                    if not time_ok() or calls_made >= budget:
                        break
                    chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    y = x + sigma_current * (chol @ rng.standard_normal(dim))
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    new_points.append(y)
                    new_fvals.append(fy)

                if len(new_points) == 0:
                    break

                idx = np.argsort(new_fvals)
                new_points = [new_points[i] for i in idx]
                new_fvals = [new_fvals[i] for i in idx]

                # Update best
                if new_fvals[0] < best_f:
                    best_f = new_fvals[0]
                    best_x = new_points[0].copy()

                points = new_points
                fvals = new_fvals

        # If we still have budget, do some local refinement
        if best_x is not None and time_ok() and calls_made < budget:
            # Simple local search
            step = 0.1
            for _ in range(min(50, budget - calls_made)):
                if not time_ok() or calls_made >= budget:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        y = best_x.copy()
                        y[d] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is None:
                            break
                        if fy < best_f:
                            best_f = fy
                            best_x = y
                            improved = True
                    if calls_made >= budget:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
