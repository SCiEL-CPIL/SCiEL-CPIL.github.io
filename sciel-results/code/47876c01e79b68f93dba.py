"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with restarts for BBOB-style f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Run CMA-ES from x0 with initial sigma0, using up to budget calls."""
        n = self.dim
        calls = 0
        best_x = x0.copy()
        best_f = np.inf

        # CMA-ES parameters
        mu = max(1, n // 2)
        lambda_ = 4 * n
        weights = np.zeros(lambda_)
        log_w = np.log(np.arange(1, lambda_ + 1))
        weights = log_w / np.sum(log_w)
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        x = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)

        while calls < budget:
            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            n_sel = min(mu, len(offspring))
            selected = offspring[:n_sel]

            # Compute mean of selected
            w_sel = weights[:n_sel]
            w_sum = np.sum(w_sel)
            x_new = np.zeros(n)
            for i, (yi, _) in enumerate(selected):
                x_new += w_sel[i] * yi
            x_new /= w_sum

            # Use current B, D for inverse sqrt(C)
            # y = B @ (D * z) => C = B @ diag(D^2) @ B.T
            # inv_sqrt(C) = B @ diag(1/D) @ B.T
            inv_sqrt_C = B @ np.diag(1.0 / D) @ B.T

            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (x_new - x))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (inv_sqrt_C @ (x_new - x))

            # Update covariance matrix
            h_sigma = 1.0 if np.linalg.norm(pc) ** 2 / (1 - (1 - c_c) ** 2) < (1 + 4 / n) * (2 - c_c) else 0.0
            C = (1 - c_1 - c_mu + c_1 * h_sigma) * C
            C += c_1 * (np.outer(pc, pc) - C)
            rank_mu = np.zeros((n, n))
            for i, (yi, _) in enumerate(selected):
                ydiff = inv_sqrt_C @ (yi - x)
                rank_mu += w_sel[i] * np.outer(ydiff, ydiff)
            C += c_mu * rank_mu
            if h_sigma == 0:
                C -= c_mu * (1 / c_c - 1) * np.outer(pc, pc)

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            C = B @ np.diag(eigvals) @ B.T

            x = x_new

            # Convergence check: if sigma too small or no progress
            if sigma < 1e-8:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        # Strategy: multiple restarts with different initial points
        # First, try center with reasonable sigma
        # Then try a few random points if budget allows

        # Initial exploration: evaluate center and a few random points
        x_center = np.zeros(n)
        fx_center = func(x_center)
        calls_made += 1
        best_f = fx_center
        best_x = x_center.copy()

        # Try a few random points to get a better starting point
        n_initial = min(10, budget // 5)
        initial_points = [x_center]
        for _ in range(n_initial):
            if calls_made >= budget:
                break
            x_rand = rng.uniform(-4, 4, n)
            fx_rand = func(x_rand)
            calls_made += 1
            initial_points.append(x_rand)
            if fx_rand < best_f:
                best_f = fx_rand
                best_x = x_rand.copy()

        # Sort initial points by fitness
        initial_points.sort(key=lambda x: func(x) if False else 0)  # We need to re-evaluate or store
        # Actually, let's just pick the best initial point
        # Re-evaluate is wasteful; we already have best_x, best_f

        # Run CMA-ES from the best initial point
        remaining = budget - calls_made
        if remaining > 0:
            x_start = best_x.copy()
            sigma_start = 1.5  # Reasonable initial step for [-5,5]^5
            bx, bf, ncalls = self._cma_es(func, x_start, sigma_start, remaining, rng)
            calls_made += ncalls
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # If we have budget left, try another restart from a different point
        remaining = budget - calls_made
        if remaining > 50 and best_f > -50:  # Only if not already very good
            # Pick a second-best initial point or random
            x_start2 = rng.uniform(-4, 4, n)
            sigma_start2 = 2.0
            bx2, bf2, ncalls2 = self._cma_es(func, x_start2, sigma_start2, remaining, rng)
            calls_made += ncalls2
            if bf2 < best_f:
                best_f = bf2
                best_x = bx2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
