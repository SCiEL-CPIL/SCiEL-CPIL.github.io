"""Multi-start (1+1)-ES with momentum for Lunacek bi-Rastrigin (BBOB f24)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with momentum for bi-modal functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def es_search(x0, sigma0, momentum_decay=0.9, max_iter=None):
            nonlocal best_x, best_fx, calls
            x = x0.copy()
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

            sigma = sigma0
            momentum = np.zeros(dim)
            no_improve = 0
            max_no_improve = 30 * dim

            while calls < budget and time.time() < deadline:
                if max_iter is not None and max_iter <= 0:
                    break
                if max_iter is not None:
                    max_iter -= 1

                # Generate candidate with momentum
                step = rng.normal(0, sigma, dim)
                new_x = x + momentum + step
                new_x = np.clip(new_x, -5, 5)

                fy = func(new_x)
                calls += 1

                if fy < fx:
                    # Success: update momentum and x
                    momentum = momentum_decay * momentum + (1 - momentum_decay) * (new_x - x)
                    x = new_x
                    fx = fy
                    no_improve = 0
                    # Slightly increase sigma on success (optional)
                    sigma *= 1.05
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                else:
                    # Failure: shrink sigma, keep momentum
                    no_improve += 1
                    sigma *= 0.9
                    if no_improve > max_no_improve:
                        break

                # Safety: if sigma too small, break
                if sigma < 1e-8:
                    break

        # Multi-start: try several different initial points
        # Lunacek has a global min near center and a local min offset
        # Try center and several random points

        # Start 1: center
        x0 = np.zeros(dim)
        es_search(x0, sigma0=1.0, max_iter=budget // 4)

        # Start 2-5: random points
        for _ in range(4):
            if calls >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-3, 3, dim)
            es_search(x0, sigma0=0.5, max_iter=budget // 8)

        # If budget remains, do one more long run from best
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining = budget - calls
            if remaining > 50:
                es_search(best_x, sigma0=0.1, max_iter=remaining)

        return best_x, best_fx
# EVOLVE-BLOCK-END
