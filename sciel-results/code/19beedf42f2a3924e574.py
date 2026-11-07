"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, start_x, sigma_init, budget, deadline, global_best):
        """Run one CMA-ES instance. Returns (best_x, best_fx, calls_made)."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        x = start_x.copy().astype(float)
        sigma = sigma_init
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        global_best[0] = min(global_best[0], best_fx)

        # Track for stagnation detection
        last_improvement_iter = 0
        fx_history = [best_fx]

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break
            gen += 1

            # Sample population from N(x, sigma^2 * C)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)

            Z = np.random.randn(lam, dim)
            proj = Z @ eigvecs
            scaled = proj * sqrt_eigvals[None, :]
            Y = x[None, :] + sigma * (scaled @ eigvecs.T)
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()
                    last_improvement_iter = gen
                    global_best[0] = min(global_best[0], best_fx)

            # Check if we had enough evaluations
            n_evaluated = np.sum(np.isfinite(F))
            if n_evaluated < 2:
                break

            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * Y_sorted[i]
            x_new = np.clip(x_new, -5, 5)

            # Update evolution paths
            y_diff = x_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_diff / sigma)

            # Compute H (outer product sum)
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y_sorted[i] - x) / sigma
                H += weights[i] * np.outer(y_i, y_i)

            # h_sigma correction
            # Use a simpler approximation to avoid numerical issues
            h_sigma = 1.0
            if gen > 1:
                # Simple check: if p_sigma norm is large relative to expected
                expected_norm = np.sqrt(mu_eff)
                if np.linalg.norm(p_sigma) > 1.4 * expected_norm:
                    h_sigma = 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_diff / sigma)

            # Update C
            C = (1 + c_1 + c_mu - c_1 - mu_eff * c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) - C) + \
                c_mu * H

            # Ensure symmetry and PSD
            C = (C + C.T) / 2
            eigvals2, eigvecs2 = np.linalg.eigh(C)
            eigvals2 = np.maximum(eigvals2, 1e-12)
            C = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new

            # Stagnation check: if no improvement for many generations
            if gen - last_improvement_iter > 3 * dim and sigma < 0.01:
                break

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        dim = self.dim
        total_budget = self.budget

        # Track global best
        global_best = [np.inf]

        # Strategy: Multiple restarts with different initial points
        # For f19 (Griewank-Rosenbrock), the landscape is relatively smooth
        # but can have multiple local minima due to the rotation/shift.

        # First, do a quick random scan to find a good starting point
        # Use ~10% of budget for initialization
        init_budget = max(10, total_budget // 10)

        # Start with center
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_fx = f0
        calls = 1
        global_best[0] = best_fx

        # Try several random points to find a better starting point
        # Use a mix of uniform and Gaussian samples for diverse exploration
        rng = np.random.default_rng(42)
        n_init = min(init_budget - 1, 25)
        for i in range(n_init):
            if calls >= total_budget or time.time() > deadline:
                break
            if i < n_init // 2:
                # Uniform samples across the box
                x_try = rng.uniform(-4, 4, size=dim)
            else:
                # Gaussian samples centered at origin with moderate spread
                x_try = rng.normal(0, 1.5, size=dim)
                x_try = np.clip(x_try, -4, 4)
            f_try = func(x_try)
            calls += 1
            if f_try < best_fx:
                best_fx = f_try
                best_x = x_try.copy()
                global_best[0] = best_fx

        # Now run CMA-ES with restarts
        # Split remaining budget among restarts
        remaining = total_budget - calls
        if remaining <= 0:
            return best_x, best_fx

        # Use 3 restarts with adaptive sigma
        n_restarts = 3
        # Allocate budget: first restart gets more, subsequent get less
        budget_alloc = [int(remaining * 0.5), int(remaining * 0.3), int(remaining * 0.2)]
        # Ensure last restart gets at least lam evaluations
        lam = 4 + int(3 * np.log(dim))
        if budget_alloc[-1] < lam and sum(budget_alloc) >= 2 * lam:
            budget_alloc[-2] -= (lam - budget_alloc[-1])
            budget_alloc[-1] = lam

        all_best_x = [best_x.copy()]
        all_best_fx = [best_fx]

        for r in range(n_restarts):
            if calls >= total_budget or time.time() > deadline:
                break

            # For first restart, use current best with larger sigma for exploration
            if r == 0:
                start_x = best_x.copy()
                sigma_init = 1.5
            else:
                # Perturb from best found so far with decreasing perturbation
                perturb_scale = 0.8 / (r + 1)
                start_x = best_x + perturb_scale * np.random.randn(dim)
                start_x = np.clip(start_x, -5, 5)
                sigma_init = 0.8 / (r + 1)

            rb = min(budget_alloc[r], total_budget - calls)
            if rb < lam:
                break

            bx, bfx, ncalls = self._cma_es(func, start_x, sigma_init, rb, deadline, global_best)
            calls += ncalls

            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()

            all_best_x.append(bx)
            all_best_fx.append(bfx)

        # Final polish: use remaining budget for high-precision local search
        if calls < total_budget and time.time() < deadline:
            polish_budget = total_budget - calls
            if polish_budget >= lam:
                # Run a final CMA-ES with very small sigma for fine-tuning
                sigma_polish = max(0.01, 0.1 * (best_fx - global_best[0] + 1e-8) ** 0.5)
                bx, bfx, ncalls = self._cma_es(func, best_x.copy(), sigma_polish, 
                                               polish_budget, deadline, global_best)
                calls += ncalls
                if bfx < best_fx:
                    best_fx = bfx
                    best_x = bx.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
