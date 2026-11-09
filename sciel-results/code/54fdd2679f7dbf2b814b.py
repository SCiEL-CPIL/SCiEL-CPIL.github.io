"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                f = 1.0
                r = 0.0
                j = i + 1
                while j > 0:
                    j, rem = divmod(j, base)
                    f /= base
                    r += rem * f
                points[i, d] = r
        return points

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Simple CMA-ES for a single run."""
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        sigma = sigma0

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        step_size = 1.0 / np.sqrt(np.diag(C))

        for _ in range(budget):
            if time.time() >= deadline:
                break

            # Sample lambda_ points
            ys = []
            for _ in range(lambda_):
                y = rng.multivariate_normal(np.zeros(dim), C)
                ys.append(y)
            ys = np.array(ys)

            # Evaluate
            xs = (x + sigma * ys).T
            fxs = np.array([func(np.clip(xi, -5, 5)) for xi in xs])

            # Sort
            idx = np.argsort(fxs)
            ys = ys[idx]
            fxs = fxs[idx]

            # Weighted recombination
            y_w = (weights[:, None] * ys[:mu]).sum(axis=0)

            # Update sigma (CSA)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / step_size - 1) * h_sigma)
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Update C
            if h_sigma:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma
            else:
                p_c = (1 - c_c) * p_c

            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff)

            C = (1 - c_c * (2 - c_c) * mu_eff * h_c - c_c * (2 - c_c) * mu_eff + c_c * (2 - c_c) * mu_eff * h_c) * C
            # Simpler update:
            C = (1 - c_c) * C + c_c * (np.outer(y_w, y_w) / (sigma ** 2) * h_c + (2 * np.sqrt(mu_eff) / (dim + 1) - 1) * C * (1 - h_sigma))

            # Ensure positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 0.1 * np.eye(dim)

            # Update mean
            x_new = x + sigma * y_w
            x_new = np.clip(x_new, -5, 5)
            fx_new = func(x_new)

            if fx_new < fx:
                x, fx = x_new, fx_new

        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Multi-start strategy with Halton sequence for initial points
        n_starts = min(10, budget // 50)
        halton_points = self._halton(n_starts, dim)
        initial_points = halton_points * 10 - 5  # Map to [-5, 5]

        # Add a few random points
        n_random = min(5, budget // 100)
        random_points = rng.uniform(-5, 5, (n_random, dim))
        all_starts = np.vstack([initial_points, random_points])

        # Evaluate all starting points
        start_vals = []
        for x in all_starts:
            if time.time() >= deadline:
                break
            fx = func(np.clip(x, -5, 5))
            start_vals.append(fx)

        if not start_vals:
            return None, None

        # Sort by fitness
        order = np.argsort(start_vals)

        # Allocate budget to starts: more to better starts
        remaining_budget = budget - len(start_vals)
        if remaining_budget <= 0:
            best_idx = order[0]
            return all_starts[best_idx], start_vals[best_idx]

        # Allocate budget proportionally to rank (better gets more)
        ranks = np.argsort(order)  # Inverse permutation
        weights = 1.0 / (1.0 + ranks)
        weights /= weights.sum()
        allocations = np.round(weights * remaining_budget).astype(int)
        # Ensure at least 10 evaluations per start if possible
        for i in range(len(allocations)):
            if allocations[i] < 10 and remaining_budget > 0:
                allocations[i] = min(10, remaining_budget)

        best_x = all_starts[order[0]].copy()
        best_f = start_vals[order[0]]

        for i in order:
            if time.time() >= deadline:
                break
            alloc = allocations[i]
            if alloc <= 0:
                continue

            x0 = all_starts[i].copy()
            sigma0 = 1.0  # Initial step size

            x_res, f_res = self._cma_es(func, x0, sigma0, alloc, rng, deadline)

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
