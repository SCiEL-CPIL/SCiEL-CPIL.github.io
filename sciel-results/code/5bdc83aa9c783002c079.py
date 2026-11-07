"""Rewritten optimizer: CMA-ES with adaptive population, restarts, and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive lambda, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial sampling to get a rough sense of the landscape
        # Try center and a few random points
        init_points = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
        ]
        rng_init = np.random.default_rng(42)
        for _ in range(10):
            init_points.append(rng_init.uniform(-4, 4, dim))

        for pt in init_points:
            if check_deadline() or calls_made >= total_budget:
                break
            eval_point(pt)

        # Phase 2: CMA-ES runs with restarts
        # We'll run multiple CMA-ES instances, each with a portion of the remaining budget
        # Strategy: run CMA-ES from the current best, with decreasing sigma, restart if stuck

        remaining_budget = total_budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # CMA-ES parameters
        # Use a population size that's reasonable for dim=5
        # lambda = 4 + floor(3 * log(dim)) ≈ 4 + 3*1.6 = 4+4 = 8, let's use 12 for better exploration
        lambda_pop = 12

        # We'll run CMA-ES for up to a certain number of generations, then restart
        max_generations_per_run = 50
        max_restarts = 10

        for restart in range(max_restarts):
            if check_deadline() or calls_made >= total_budget:
                break

            # Start from current best with some perturbation for later restarts
            if restart == 0:
                x_mean = best_x.copy()
                sigma = 1.0
            else:
                # Perturb the best point for diversity
                rng_restart = np.random.default_rng(1000 + restart)
                x_mean = best_x + rng_restart.normal(0, 0.5, dim)
                x_mean = np.clip(x_mean, -5.0, 5.0)
                sigma = 0.5

            # CMA-ES state
            C = np.eye(dim)
            # Weights and mu
            mu = lambda_pop // 2
            w = np.ones(mu)
            w[0] = 2.0
            w /= w.sum()
            mu_eff = 1.0 / (w @ w)

            # Step size control
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs
            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
            cmu = (2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)) * \
                  (min(1, c1) - c1)

            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Precompute sqrt of C for sampling (will be updated)
            # For simplicity, use Cholesky or eigen decomposition each step
            # Actually, for dim=5, we can use SVD or just random directions with adaptation

            # Simpler approach: use a basic CMA-ES with fixed correlation matrix initially
            # and adapt sigma only, plus some random restarts. This is more robust.

            # Let's use a simpler but effective approach:
            # Adaptive sigma with random directions, plus occasional large jumps

            gen = 0
            no_improve_count = 0
            local_best_f = best_f
            local_best_x = best_x.copy()

            while gen < max_generations_per_run:
                if check_deadline() or calls_made >= total_budget:
                    break

                # Generate lambda_pop candidates
                candidates = []
                rng_gen = np.random.default_rng(np.random.randint(0, 2**31))

                for i in range(lambda_pop):
                    if i == 0:
                        # First candidate: small step from mean
                        direction = rng_gen.standard_normal(dim)
                        candidate = x_mean + sigma * direction
                    elif i == 1:
                        # Second candidate: medium step
                        direction = rng_gen.standard_normal(dim)
                        candidate = x_mean + sigma * 2.0 * direction
                    else:
                        # Rest: standard normal steps
                        direction = rng_gen.standard_normal(dim)
                        candidate = x_mean + sigma * direction

                    candidate = np.clip(candidate, -5.0, 5.0)
                    candidates.append(candidate)

                # Evaluate all candidates
                values = []
                for cand in candidates:
                    if check_deadline() or calls_made >= total_budget:
                        break
                    val = eval_point(cand)
                    if val is None:
                        break
                    values.append((val, cand))

                if not values:
                    break

                # Sort by fitness
                values.sort(key=lambda x: x[0])
                best_val, best_cand = values[0]

                if best_val < local_best_f:
                    local_best_f = best_val
                    local_best_x = best_cand.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Update mean and sigma based on success
                # Use the top mu candidates
                top_mu = values[:min(mu, len(values))]
                new_mean = np.zeros(dim)
                for w_i, (v, c) in zip(w[:len(top_mu)], top_mu):
                    new_mean += w_i * c

                # Adaptive sigma: if we made progress, increase; else decrease
                if best_val < local_best_f + 1e-8:  # progress made
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                sigma = max(sigma, 0.01)
                sigma = min(sigma, 5.0)

                # Update mean: move towards the best candidates
                x_mean = 0.5 * x_mean + 0.5 * new_mean
                x_mean = np.clip(x_mean, -5.0, 5.0)

                gen += 1

                # Check for stagnation
                if no_improve_count > 20:
                    break

            # After CMA-ES run, if we improved, continue with next restart
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # If we've used most of the budget, stop
            if calls_made >= total_budget:
                break

            # If no improvement after a full run, and we've done several restarts, stop
            if restart > 0 and local_best_f >= best_f - 1e-10:
                # No improvement, try a different strategy
                if restart >= 5:
                    break

        # Final: if we have budget left, do a local refinement around best
        if not check_deadline() and calls_made < total_budget:
            # Simple local search with decreasing step
            step = 0.1
            rng_final = np.random.default_rng(999)
            while step > 0.001 and not check_deadline() and calls_made < total_budget:
                improved = False
                for _ in range(10):
                    if check_deadline() or calls_made >= total_budget:
                        break
                    direction = rng_final.standard_normal(dim)
                    direction /= np.linalg.norm(direction)
                    candidate = best_x + step * direction
                    candidate = np.clip(candidate, -5.0, 5.0)
                    val = eval_point(candidate)
                    if val is not None and val < best_f:
                        improved = True
                if not improved:
                    step *= 0.5
                else:
                    step *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
