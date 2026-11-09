"""CMA-ES with adaptive restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for multimodal Weierstrass in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # We'll run CMA-ES with multiple restarts
        # Each restart gets a portion of the budget
        # Strategy: first run gets most budget, subsequent ones get less
        # but we adapt based on improvement

        def cma_es(x0, sigma0, max_calls, rng):
            """Run CMA-ES from x0 with initial sigma sigma0, using at most max_calls evaluations."""
            nonlocal total_calls, best_x, best_f

            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # Weights
            mu = lambda_ // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # Initial state
            x = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            cs = np.sqrt((n + 1) * c_sigma * (2 - c_sigma))
            chi_n = n * (1 - 1.0 / (4 * n) - 1.0 / (20 * n ** 2))

            # Initial evaluations
            x = np.clip(x, -5, 5)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Generate initial population
            def generate_population():
                nonlocal total_calls
                B = np.linalg.cholesky(C)
                # Generate lambda_ standard normal vectors
                N = rng.standard_normal((lambda_, n))
                y = N
                # Transform to get candidates
                candidates = np.array([np.clip(x + sigma * (B @ y[i].T).T, -5, 5) for i in range(lambda_)])
                fvals = np.array([func(candidates[i]) for i in range(lambda_)])
                total_calls += lambda_
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = candidates[min_idx].copy()
                return candidates, fvals, y

            # Initial population
            if total_calls + lambda_ <= max_calls:
                candidates, fvals, y = generate_population()
                # Sort
                order = np.argsort(fvals)
                y_sorted = y[order]
                fvals_sorted = fvals[order]
                x_sorted = candidates[order]

                # Update
                xnew = np.sum(w[:, None] * x_sorted[:mu], axis=0)
                # Step size control
                ys = (B @ (xnew - x).T).T / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ys
                hsig = (np.sqrt(np.sum(ps ** 2)) / np.sqrt(1 - (1 - c_sigma) ** 20) / chi_n - 1) ** 2 < 1.4 + 2 / (n + 1)
                if hsig:
                    pc = (1 - c_c) * pc + c_c * (xnew - x) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * np.sum(w[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
                # Ensure symmetry
                C = (C + C.T) / 2
                x = xnew
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Main loop
            while total_calls + lambda_ <= max_calls:
                if time.time() > deadline - 5:
                    break
                # Generate population
                B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                N = rng.standard_normal((lambda_, n))
                candidates = np.array([np.clip(x + sigma * (B @ N[i].T).T, -5, 5) for i in range(lambda_)])
                fvals = np.array([func(candidates[i]) for i in range(lambda_)])
                total_calls += lambda_

                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = candidates[min_idx].copy()

                order = np.argsort(fvals)
                y_sorted = N[order]
                x_sorted = candidates[order]

                xnew = np.sum(w[:, None] * x_sorted[:mu], axis=0)
                ys = (B @ (xnew - x).T).T / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ys
                hsig = (np.sqrt(np.sum(ps ** 2)) / np.sqrt(1 - (1 - c_sigma) ** 20) / chi_n - 1) ** 2 < 1.4 + 2 / (n + 1)
                if hsig:
                    pc = (1 - c_c) * pc + c_c * (xnew - x) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * np.sum(w[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
                C = (C + C.T) / 2
                x = xnew
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Termination conditions
                if sigma < 1e-12:
                    break
                if np.max(np.abs(C)) < 1e-15:
                    break
                # Check if progress is minimal
                if lambda_ > 0 and np.std(fvals[:mu]) < 1e-12:
                    break

            return best_x, best_f

        # Multi-start strategy
        # First, try a few random starting points with CMA-ES
        rng = np.random.default_rng(42)

        # Allocate budget: give more to earlier restarts
        # First restart: 60% of budget, second: 25%, third: 15%
        restarts = [
            (0.60, 3.0),
            (0.25, 2.0),
            (0.15, 1.5),
        ]

        for frac, sigma0 in restarts:
            if total_calls >= budget:
                break
            if time.time() > deadline - 10:
                break
            max_calls = min(int(frac * budget), budget - total_calls)
            if max_calls < 10:
                break

            # Random start in [-4, 4]^5
            x0 = rng.uniform(-4, 4, dim)
            cma_es(x0, sigma0, max_calls, rng)

        # If we still have budget, do a local (1+1)-ES polish from the best point
        if total_calls < budget and best_x is not None and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while total_calls < budget:
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-10:
                        break

        # Final check: make sure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
