"""Simulated Annealing optimizer for BBOB-style f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initial point: try a few random starts, pick best
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Try a few random initializations to find a better starting point
        for _ in range(10):
            x_try = rng.uniform(-4, 4, self.dim)
            fx_try = func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
                x = x_try
                fx = fx_try

        # Simulated Annealing parameters
        T0 = 50.0          # initial temperature
        T_end = 0.01       # final temperature
        n_iter = self.budget - 11  # remaining budget after init
        # Geometric cooling: T_k = T0 * (T_end/T0)^(k/n_iter)
        alpha = (T_end / T0) ** (1.0 / max(n_iter, 1))

        step_size = 3.0    # initial step size (large for exploration)
        step_decay = 0.9995  # slow decay of step size

        T = T0
        x = best_x.copy()
        fx = best_fx

        for i in range(n_iter):
            if time.time() > deadline:
                break

            # Generate neighbor
            dx = rng.standard_normal(self.dim)
            y = np.clip(x + step_size * dx, -5, 5)
            fy = func(y)

            if fy < fx:
                # Accept improvement
                x, fx = y, fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
            else:
                # Accept worse with probability exp(-(fy-fx)/T)
                delta = fy - fx
                if T > 1e-12:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                if rng.random() < prob:
                    x, fx = y, fy

            # Cool down
            T *= alpha
            # Slowly reduce step size
            step_size *= step_decay

        return best_x, best_fx
# EVOLVE-BLOCK-END
