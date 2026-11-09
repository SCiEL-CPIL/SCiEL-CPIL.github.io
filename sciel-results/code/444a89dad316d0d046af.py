"""Rewritten optimizer: CMA-ES with adaptive population and restarts for Rastrigin rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 12 and time-budgeted restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
        max_sigma = 10.0
        min_sigma = 1e-12

        # We'll run CMA-ES with restarts. Each restart uses a fresh CMA-ES.
        # Budget is split across restarts.
        # For Rastrigin, multiple restarts help escape local minima.

        # Strategy: run CMA-ES for a portion of the budget, then restart if no improvement.
        # But we also need to be careful about total calls.

        # Let's use a simpler approach: run CMA-ES with the full budget, but with
        # a mechanism to detect stagnation and restart.

        # Actually, let's implement a proper CMA-ES with restarts.
        # Each restart: initialize CMA-ES, run for some generations, check for progress.

        # Given the complexity, let's use a well-tuned CMA-ES with a single run
        # but with a larger population and adaptive step size.

        # CMA-ES implementation
        def cma_es(func, budget_remaining, x0, sigma0, max_time):
            nonlocal total_calls, best_x, best_f

            n = dim
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
            mu = lam // 2

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Step size control parameters
            cs = (mueff + 1) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = (2 - mueff / (n + 1) ** 1.5 + 2 * weights[0]) / ((n + 2) ** 1.5 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + weights[0]) / ((n + 2) ** 2 + mueff))

            # Initialize
            x = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            Cinv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial evaluations
            best_local_x = x.copy()
            best_local_f = func(x)
            total_calls += 1
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()

            # Generate initial population
            def generate_population():
                nonlocal total_calls
                # Sample from N(x, sigma^2 * C)
                # Use Cholesky decomposition for sampling
                L = np.linalg.cholesky(C)
                z = np.random.default_rng().standard_normal((lam, n))
                pop = x[:, None] + sigma * (z @ L.T)
                # Clip to bounds
                pop = np.clip(pop, -5, 5)

                # Evaluate
                fvals = []
                for i in range(lam):
                    if total_calls >= budget_remaining:
                        fvals.append(np.inf)
                    else:
                        fi = func(pop[i])
                        total_calls += 1
                        fvals.append(fi)
                        if fi < best_local_f:
                            best_local_f = fi
                            best_local_x = pop[i].copy()
                            if fi < best_f:
                                best_f = fi
                                best_x = pop[i].copy()
                return pop, np.array(fvals)

            pop, fvals = generate_population()

            # Main loop
            while total_calls < budget_remaining:
                if time.time() > max_time:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * pop_sorted[i] for i in range(mu))

                # Update step size
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Cinv @ (x - x_old))
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (total_calls)))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, min_sigma, max_sigma)

                # Update covariance
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x - x_old) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C)
                for i in range(mu):
                    y_i = (pop_sorted[i] - x_old) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    Cinv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    Cinv = np.eye(n)

                # Generate next population
                pop, fvals = generate_population()

                # Check for stagnation
                if fvals_sorted[0] >= best_local_f - 1e-10:
                    # No improvement in this generation
                    pass

            return best_local_x, best_local_f

        # Run CMA-ES with multiple restarts
        # Strategy: split budget into chunks, restart if no significant improvement
        n_restarts = 3
        budget_per_restart = budget // n_restarts

        rng = np.random.default_rng(42)

        for restart in range(n_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize around best found so far, or random
            if best_x is not None and restart > 0:
                # Perturb around best
                x0 = best_x + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0

            cma_es(func, budget - total_calls, x0, sigma0, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
