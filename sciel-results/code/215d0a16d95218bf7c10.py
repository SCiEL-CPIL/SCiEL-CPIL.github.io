"""Multi-start (1+1)-ES with momentum for BBOB f24 (Lunacek bi-Rastrigin).

Strategy:
  1. Multiple independent restarts to handle the bimodal nature of Lunacek bi-Rastrigin.
  2. (1+1)-ES with step-size adaptation (CSA) for efficient local search.
  3. Momentum to help navigate the Rastrigin landscape's many local minima.
  4. Lightweight per-evaluation cost allows using full budget effectively.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with momentum for 5D box-constrained problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = float('inf')
        evals = 0

        # Number of restarts: use 5 restarts, each with ~100 evaluations
        num_restarts = 5
        budget_per_restart = total_budget // num_restarts

        for restart in range(num_restarts):
            if evals >= total_budget:
                break
            if time.time() > deadline:
                break

            # Initialize this restart
            # First restart from center, others from random points
            if restart == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-4, 4, size=n)

            fx = func(x)
            evals += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES parameters
            sigma = 2.0  # Initial step size
            momentum = np.zeros(n)
            momentum_alpha = 0.3  # Momentum coefficient
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            c_sigma = 1.0 / (n + 3)  # Step-size adaptation rate
            d_sigma = 1 + 2 * max(0, 1 / (n + 1)) + c_sigma

            # Local budget for this restart
            local_budget = budget_per_restart
            local_evals = 0
            stagnation_count = 0
            max_stagnation = 20  # Restart if no improvement for 20 evals

            while local_evals < local_budget and evals < total_budget:
                if time.time() > deadline:
                    break

                # Generate candidate
                z = rng.standard_normal(n)
                # Apply momentum
                direction = z + momentum_alpha * momentum
                # Normalize to maintain step size
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir * chi_n
                else:
                    direction = z * chi_n

                x_candidate = x + sigma * direction
                x_candidate = np.clip(x_candidate, -5, 5)

                fx_candidate = func(x_candidate)
                evals += 1
                local_evals += 1

                if fx_candidate < fx:
                    # Success: update position and momentum
                    delta = x_candidate - x
                    momentum = momentum_alpha * momentum + (1 - momentum_alpha) * delta / sigma
                    x = x_candidate
                    fx = fx_candidate
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(delta) / (sigma * chi_n) - 1))
                    sigma = np.clip(sigma, 1e-8, 10.0)
                    stagnation_count = 0

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: reduce step size slightly
                    sigma *= np.exp(-c_sigma / (2 * d_sigma))
                    sigma = np.clip(sigma, 1e-8, 10.0)
                    stagnation_count += 1

                # Early termination if stagnated
                if stagnation_count >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
