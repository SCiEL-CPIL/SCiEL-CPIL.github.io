"""Rewritten optimizer: multi-start Nelder-Mead + CMA-ES hybrid for f22."""
import time
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, rng):
        """Nelder-Mead simplex method with budget limit."""
        n = self.dim
        # Build initial simplex
        simplex = [np.array(x0, dtype=float)]
        fx = [func(simplex[0])]
        for i in range(n):
            dx = np.zeros(n)
            step = 0.5 if fx[0] > 0 else 0.2
            dx[i] = step
            simplex.append(simplex[0] + dx)
            fx.append(func(simplex[-1]))

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        calls = n + 1
        limit = budget

        while calls < limit:
            # Sort by function value
            order = np.argsort(fx)
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fx[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < fx[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fx[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fx[-1] = f_r
            elif f_r < fx[-2]:
                simplex[-1] = x_r
                fx[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c <= min(f_r, f_worst):
                    simplex[-1] = x_c
                    fx[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx[i] = func(simplex[i])
                        calls += 1
                    if calls >= limit:
                        break

        # Return best
        best_idx = int(np.argmin(fx))
        return simplex[best_idx], fx[best_idx]

    def _cma_es(self, func, x0, budget, rng, sigma0=1.0):
        """CMA-ES implementation."""
        n = self.dim
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < budget:
            # Generate population
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = B @ (D @ z)
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                pop.append((xi, func(xi)))
                calls += 1
                if calls >= budget:
                    break

            # Sort
            pop.sort(key=lambda t: t[1])
            selected = [pop[i][0] for i in range(min(mu, len(pop)))]
            f_selected = [pop[i][1] for i in range(min(mu, len(pop)))]

            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

            # Weighted mean
            x_new = np.sum(np.array([w * s for w, s in zip(weights, selected)]), axis=0)

            # Update evolution paths
            y_mean = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            norm_ps = np.linalg.norm(p_sigma)
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - 1)))) < \
                      (1 + 2 / (n + 1)) * np.sqrt(1 - c_sigma)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i, w in enumerate(weights):
                diff = (selected[i] - x) / sigma
                rank_mu += w * np.outer(diff, diff)

            C = (1 + c_1 * (delta_h - 1) + c_mu) * C - c_1 * rank_1 - c_mu * rank_mu

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(2 * n) - 1))
            sigma = max(sigma, 1e-10)

            # Update mean
            x = x_new

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Phase 1: Multi-start Nelder-Mead for rapid exploration
        # Use ~60% of budget for exploration
        nm_budget = int(budget * 0.55)
        n_starts = min(20, max(5, nm_budget // 15))
        per_start_budget = nm_budget // n_starts

        best_x = None
        best_f = np.inf

        for i in range(n_starts):
            if time.time() > deadline:
                break

            # Diverse starting points
            if i == 0:
                x0 = np.zeros(n)
            elif i == 1:
                x0 = np.array([2.0, -2.0, 2.0, -2.0, 2.0][:n])
            elif i == 2:
                x0 = np.array([-2.0, 2.0, -2.0, 2.0, -2.0][:n])
            else:
                x0 = rng.uniform(-4, 4, n)

            x, f = self._nelder_mead(func, x0, per_start_budget, rng)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: CMA-ES refinement from best point
        remaining = budget - nm_budget
        if remaining > 10 and best_x is not None:
            if time.time() < deadline:
                # Estimate good sigma based on current basin
                sigma0 = 1.0
                x_cma, f_cma = self._cma_es(func, best_x, remaining, rng, sigma0)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()

        return best_x, best_f
