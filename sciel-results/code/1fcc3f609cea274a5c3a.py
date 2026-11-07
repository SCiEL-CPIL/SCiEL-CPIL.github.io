"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_run(self, func, x0, f0, max_calls, rng, deadline):
        """Run CMA-ES from x0. Returns (best_x, best_f, calls_used)."""
        n = self.dim
        calls = 1  # f0 already evaluated

        # CMA-ES parameters for small n
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (n + 1.7) / (n + 5 + 2 * (n / (lam - 2)))
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1) + c_sigma
        c_1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + lam)
        c_mu = min(1 - c_1, (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + lam))

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) - 1 / (25 * n ** 2))

        x_mean = x0.copy().astype(float)
        sigma = 1.0
        B = np.eye(n)
        D = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)

        best_x = x0.copy()
        best_f = f0

        for gen in range(max_calls - 1):
            if time.time() > deadline:
                break
            if calls >= max_calls:
                break

            # Sample population
            z = rng.standard_normal((lam, n))
            pop = x_mean[None, :] + sigma * (z @ (B * D).T)
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = np.empty(lam)
            for i in range(lam):
                fvals[i] = func(pop[i])
                calls += 1
                if calls >= max_calls:
                    # Fill remaining with large value
                    fvals[i+1:] = np.inf
                    break

            # Update best
            idx = np.argmin(fvals)
            if fvals[idx] < best_f:
                best_f = fvals[idx]
                best_x = pop[idx].copy()

            # Sort
            order = np.argsort(fvals)
            x_sorted = pop[order]
            f_sorted = fvals[order]

            # Weighted mean in z-space
            x_tilde = np.zeros(n)
            for i in range(mu):
                x_tilde += w[i] * (x_sorted[i] - x_mean) / (sigma * (B * D))
            # Actually, x_tilde should be in the z-space:
            # pop[i] = x_mean + sigma * z_i @ (B*D).T
            # So (x_sorted[i] - x_mean) / sigma = z_i @ (B*D).T
            # We need z_i, which is harder. Let's use the standard formula.

            # Standard CMA-ES update:
            # x_new_mean = x_mean + sigma * sum(w_i * (x_sorted[i] - x_mean) / sigma)
            #            = x_mean + sum(w_i * (x_sorted[i] - x_mean))
            x_new_mean = x_mean + np.sum([w[i] * (x_sorted[i] - x_mean) for i in range(mu)], axis=0)
            x_new_mean = np.clip(x_new_mean, -5, 5)

            # Step size control
            x_tilde_z = np.zeros(n)
            for i in range(mu):
                x_tilde_z += w[i] * (x_sorted[i] - x_mean) / sigma

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * x_tilde_z

            norm_p_sigma = np.linalg.norm(p_sigma)
            if norm_p_sigma > 1e-15:
                h_sigma = min(1.0, norm_p_sigma / (chi_n * np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) + 1e-15))
            else:
                h_sigma = 0.0

            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Covariance adaptation
            p_c = (1 - c_1) * p_c + np.sqrt(
                c_1 * (2 - c_1) * mu_eff
            ) * x_tilde_z

            norm_p_c = np.linalg.norm(p_c)
            if norm_p_c > 1e-15:
                h_c = min(1.0, norm_p_c / (chi_n * np.sqrt(1 - (1 - c_1) ** (gen + 1)) + 1e-15))
            else:
                h_c = 0.0

            delta_h = (1 - h_c) * c_1 * (2 - c_1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += w[i] * np.outer(x_tilde_z[i], x_tilde_z[i])

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + delta_h * np.eye(n)) + c_mu * rank_mu

            # Cholesky / eigen decomposition for B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-15, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            x_mean = x_new_mean

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280  # safety margin

        n = self.dim
        budget = self.budget

        # Multi-start with diverse initial points
        n_starts = 4
        starts = [np.zeros(n)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, n))

        best_x = None
        best_f = np.inf
        calls_made = 0

        # First, evaluate all starts
        for x0 in starts:
            if calls_made >= budget:
                break
            f0 = func(x0)
            calls_made += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Run CMA-ES from the best start with remaining budget
        remaining = budget - calls_made
        if remaining > 20 and best_x is not None and time.time() < deadline:
            x_res, f_res, c_used = self._cmaes_run(func, best_x, best_f, remaining, rng, deadline)
            calls_made += c_used
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # If we still have budget, try from other starts
        for x0 in starts[1:]:
            if calls_made >= budget or time.time() > deadline:
                break
            remaining = budget - calls_made
            if remaining < 30:
                break
            # Evaluate if not already done
            f0 = func(x0)
            calls_made += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            x_res, f_res, c_used = self._cmaes_run(func, x0, f0, remaining, rng, deadline)
            calls_made += c_used
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
