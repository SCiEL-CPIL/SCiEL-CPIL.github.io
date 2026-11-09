import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with CMA-ES refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline - 5

        # Phase 1: Diverse initial exploration
        # Use structured + random points for good coverage of [-4,4]^5
        n_starts = 30
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Axis-aligned points (±3 in each dimension)
        for i in range(n):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                start_points.append(pt.copy())

        # Diagonal points
        for val in [-2.0, 2.0]:
            start_points.append(np.full(dim, val))

        # Random points to fill up to n_starts
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-4, 4, dim))

        # Shuffle for diversity
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx[:n_starts]]

        for sp in start_points:
            if call_count >= budget - 1 or check_time():
                break
            x = np.clip(sp.copy(), -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: Adaptive (1+1)-ES from best point
        # Use ~50% of remaining budget for this phase
        remaining = budget - call_count
        es_budget = int(remaining * 0.5)

        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        stagnation = 0
        max_stagnation = 40
        improvement_threshold = 1e-10

        for _ in range(es_budget):
            if call_count >= budget - 1 or check_time():
                break

            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx - improvement_threshold:
                x = y
                fx = fy
                sigma *= 1.25
                stagnation = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.88
                stagnation += 1
                if stagnation >= max_stagnation:
                    sigma = max(0.3, min(3.0, sigma * 2.5))
                    stagnation = 0

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 5.0)

        # Phase 3: Full CMA-ES refinement
        if call_count < budget - 5 and not check_time():
            lambda_ = 4 + int(3 * np.log(n))  # 8 for n=5
            mu_ = lambda_ // 2  # 4
            weights = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (n + 4.3)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))

            x = best_x.copy()
            sigma = 0.5
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Precompute Cholesky for efficiency
            L = np.linalg.cholesky(C + 1e-10 * np.eye(n))

            while call_count < budget:
                if check_time():
                    break

                # Generate lambda_ candidates
                ys = []
                fs = []
                for _ in range(lambda_):
                    if call_count >= budget or check_time():
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    call_count += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < mu_:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])

                # Update mean
                x_old = x.copy()
                x = np.sum(weights[:mu_][:, np.newaxis] * ys_sorted[:mu_], axis=0)

                # Compute L^{-1}(x - x_old) / sigma
                diff = x - x_old
                try:
                    v = np.linalg.solve(L, diff / sigma)
                except np.linalg.LinAlgError:
                    v = np.linalg.lstsq(L, diff / sigma, rcond=None)[0]

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * v
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(n) - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 5.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * v

                # Rank-mu update
                y_diffs = ys_sorted[:mu_] - x_old
                try:
                    y_diffs_L_inv = np.linalg.solve(L, y_diffs.T).T
                except np.linalg.LinAlgError:
                    y_diffs_L_inv = np.linalg.lstsq(L, y_diffs.T, rcond=None)[0].T

                C = (1 - c_1 - c_mu) * C + \
                    c_1 * np.outer(p_c, p_c) + \
                    c_mu * np.sum(weights[i] * np.outer(y_diffs_L_inv[i], y_diffs_L_inv[i]) for i in range(mu_))

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    L = np.linalg.cholesky(C + 1e-6 * np.eye(n))

        return best_x, best_f
