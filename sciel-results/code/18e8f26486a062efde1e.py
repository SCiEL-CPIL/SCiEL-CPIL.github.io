"""Rewritten optimizer for BBOB f15 (Rastrigin rotated) with structural change."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like with Rastrigin-aware integer snapping and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = None
        best_fx = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Strategy: multiple short CMA-ES-like runs with different initializations.
        # Each run uses a population-based evolution strategy with rank-based weights
        # and adaptive step size (similar to CMA-ES but simplified for low dim).
        # Between runs, we snap coordinates to integers to exploit Rastrigin structure.

        n_restarts = 6
        calls_per_run = max(10, (budget - n_restarts * 5) // n_restarts)

        for restart in range(n_restarts):
            if not time_ok() or calls >= budget - 5:
                break

            # Initialize center: mix of random and integer-snapped points
            if restart == 0:
                center = rng.uniform(-3, 3, dim)
            elif restart == 1:
                center = np.round(rng.uniform(-3, 3, dim))
            elif restart == 2:
                center = np.sign(rng.standard_normal(dim)) * np.abs(rng.uniform(0, 3, dim))
            else:
                # Perturb best found so far
                if best_x is not None:
                    center = best_x + rng.standard_normal(dim) * 0.5
                else:
                    center = rng.uniform(-4, 4, dim)
            center = np.clip(center, -4.5, 4.5)

            # Evaluate center
            evaluate(center)
            if not time_ok() or calls >= budget - 3:
                break

            # Population-based ES from this center
            pop_size = min(15, max(8, dim + 5))
            sigma = 1.0
            max_iter = min(calls_per_run, budget - calls - 3)

            # Maintain a small population
            pop = [(center.copy(), best_fx if np.allclose(center, best_x) else float(func(center)))]
            # Re-evaluate center properly
            pop = [(center.copy(), evaluate(center) if not np.allclose(center, best_x) else best_fx)]

            for it in range(max_iter):
                if not time_ok() or calls >= budget - 2:
                    break

                # Generate offspring
                offspring = []
                for j in range(pop_size):
                    if not time_ok() or calls >= budget - 1:
                        break
                    y = center + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    offspring.append((y, fy))

                if not offspring:
                    break

                # Sort offspring
                offspring.sort(key=lambda t: t[1])

                # Update center as weighted mean of top half
                lam = len(offspring)
                top_k = max(1, lam // 2)
                weights = np.ones(top_k)
                weights[0] = 2.0  # Emphasize best
                w_sum = weights.sum()
                new_center = np.zeros(dim)
                for k in range(top_k):
                    new_center += weights[k] * offspring[k][0]
                new_center /= w_sum
                new_center = np.clip(new_center, -5.0, 5.0)

                # Adaptive sigma: increase if best improved, decrease otherwise
                old_best = best_fx
                center = new_center
                if best_fx < old_best:
                    sigma *= 1.15
                else:
                    sigma *= 0.85
                sigma = max(sigma, 1e-6)

                # Rastrigin-aware: try integer snapping of current center
                if it % 5 == 4:
                    x_int = np.round(center)
                    x_int = np.clip(x_int, -5.0, 5.0)
                    if not np.allclose(x_int, center, atol=1e-6):
                        evaluate(x_int)

            # After each run, do a few integer-snapping probes around best
            if best_x is not None and time_ok() and calls < budget - 3:
                for coord in range(dim):
                    if not time_ok() or calls >= budget - 2:
                        break
                    for delta in [-0.5, 0.5]:
                        if not time_ok() or calls >= budget - 1:
                            break
                        probe = best_x.copy()
                        probe[coord] = np.clip(np.round(probe[coord]) + delta, -5.0, 5.0)
                        evaluate(probe)

        # Final polish: coordinate-wise local search
        if best_x is not None and time_ok() and calls < budget - 2:
            sigma_small = 0.1
            for _ in range(min(20, budget - calls - 1)):
                if not time_ok() or calls >= budget - 1:
                    break
                y = np.clip(best_x + sigma_small * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                if fy < best_fx:
                    sigma_small *= 1.1
                else:
                    sigma_small *= 0.85
                    if sigma_small < 1e-8:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
