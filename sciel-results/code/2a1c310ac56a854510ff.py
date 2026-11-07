"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size and periodic local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: center is often good for Rastrigin-like shifted functions,
        # but we'll also try a few random starts and keep the best.
        # Actually, for Rastrigin with shift, the minimum is somewhere in [-4,4]^5.
        # Let's start near origin and also do a small random search to find a good basin.

        # Phase 1: Brief random search to find a promising region (5% of budget)
        n_init = max(10, budget // 20)
        x = np.zeros(dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        for i in range(n_init - 1):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
        x, fx = best_x.copy(), best_fx

        # Phase 2: (1+1)-ES with 1/5 success rule
        sigma = 1.0
        n_steps = budget - n_init
        success_window = []
        window_size = 15

        for step in range(n_steps):
            if time.time() > deadline:
                break

            # Adaptive sigma via 1/5 success rule
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window)
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.5:
                    sigma *= 0.7
                success_window = success_window[1:]

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                success_window.append(0)

            # Periodic local refinement: every 50 steps, do a few small perturbations
            if (step + 1) % 50 == 0 and step < n_steps - 10:
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    x_local = np.clip(x + 0.1 * rng.standard_normal(dim), -5, 5)
                    fx_local = func(x_local)
                    if fx_local < fx:
                        x, fx = x_local, fx_local
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
