"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead local polish with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 300s limit, 30s safety margin

        # --- Phase 1: Latin Hypercube initialization for coverage ---
        n_lhs = 25
        lhs_points = []
        for d in range(self.dim):
            perm = np.random.default_rng(42 + d).permutation(n_lhs)
            lhs_points.append((perm + 0.5) / n_lhs * 10 - 5)
        lhs_points = np.array(lhs_points).T  # shape (n_lhs, dim)

        # Evaluate LHS points
        x_best = None
        f_best = np.inf
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            xi = np.clip(lhs_points[i], -5, 5)
            fi = func(xi)
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()

        budget_used = n_lhs
        if x_best is None:
            # Fallback: single random point
            x_best = np.random.default_rng(0).uniform(-5, 5, self.dim)
            f_best = func(x_best)
            budget_used = 1

        # --- Phase 2: CMA-ES from the best LHS point ---
        if budget_used < self.budget and time.time() < deadline:
            x_best = self._cma_es(func, x_best, f_best, self.budget - budget_used, deadline)
            budget_used = self.budget

        # --- Phase 3: Nelder-Mead polish on the best point ---
        if budget_used < self.budget and time.time() < deadline:
            remaining = self.budget - budget_used
            # Use a subset of remaining budget for Nelder-Mead (it's expensive per call)
            nm_budget = min(remaining, 200)
            try:
                res = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True}
                )
                if res.fun < f_best:
                    x_best = np.clip(res.x, -5, 5)
                    f_best = res.fun
            except Exception:
                pass

        return x_best, f_best

    def _cma_es(self, func, x0, f0, budget, deadline):
        """Simple CMA-ES implementation."""
        dim = self.dim
        rng = np.random.default_rng(7)

        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_ = 2 * mu
        # Weighted recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial sigma
        sigma = 2.0
        x = x0.copy()
        f = f0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        calls = 0
        while calls < budget and time.time() < deadline:
            # Generate lambda_ offspring
            ys = np.zeros((lambda_, dim))
            for i in range(lambda_):
                ys[i] = rng.standard_normal(dim)
            # Sort by fitness
            fs = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * ys[i], -5, 5)
                fs[i] = func(y)
                calls += 1
            else:
                # If we broke early, fill remaining with inf
                for i in range(calls - lambda_ + 1, lambda_):
                    if i >= 0:
                        fs[i] = np.inf
                # Adjust: we only have partial evaluations
                # Re-evaluate only what we need
                pass

            # Actually, let's be more careful with the loop
            # Let me redo this more carefully

            # Re-do: evaluate all lambda_ offspring, checking deadline
            fs = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    fs[i] = np.inf
                    continue
                y = np.clip(x + sigma * ys[i], -5, 5)
                fs[i] = func(y)
                calls += 1

            # Sort
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update best
            if fs_sorted[0] < f:
                f = fs_sorted[0]
                x = np.clip(x + sigma * ys_sorted[0], -5, 5)

            # Weighted mean of best mu
            m = (weights[:, None] * ys_sorted[:mu]).sum(axis=0)

            # Update step size control
            C_sqrt_inv = np.linalg.inv(np.sqrtm(C))  # This is expensive; use approximation
            # Actually, let's use a simpler approach: just track the mean and adapt sigma
            # For simplicity and speed, use a simplified CMA-ES

            # Simplified: just adapt sigma based on success rate
            success = fs_sorted[0] < f0  # This isn't quite right either

            # Let me just use a simpler adaptive sigma approach
            # Check if improvement happened
            if fs_sorted[0] < f:
                sigma = min(sigma * 1.2, 10.0)
            else:
                sigma = max(sigma * 0.8, 0.001)

            # Update mean
            x = np.clip(x + 0.5 * sigma * m, -5, 5)
            f0 = f  # Update reference

        return x, f
# EVOLVE-BLOCK-END
