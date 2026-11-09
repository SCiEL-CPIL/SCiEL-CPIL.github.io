"""Particle Swarm Optimization for BBOB f11 (Discus) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """PSO with inertia, cognitive, and social components."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize swarm
        swarm_size = 20
        x = rng.uniform(-5, 5, (swarm_size, dim))
        v = rng.uniform(-1.5, 1.5, (swarm_size, dim))

        # Evaluate initial positions
        fx = np.array([func(xi) for xi in x])
        pbest_x = x.copy()
        pbest_f = fx.copy()
        gbest_idx = np.argmin(fx)
        gbest_x = x[gbest_idx].copy()
        gbest_f = fx[gbest_idx]

        # PSO parameters
        w0 = 0.9      # initial inertia
        w_final = 0.4 # final inertia
        c1 = 1.5      # cognitive
        c2 = 1.5      # social
        max_v = 2.0   # velocity clamp

        calls_made = swarm_size

        for it in range(budget - swarm_size):
            # Check time
            if time.time() > deadline:
                break

            # Linearly decrease inertia
            w = w0 + (w_final - w0) * (it / max(1, budget - swarm_size - 1))

            for i in range(swarm_size):
                # Check time per particle (coarse check)
                if time.time() > deadline:
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Update velocity
                v[i] = w * v[i] + c1 * r1 * (pbest_x[i] - x[i]) + c2 * r2 * (gbest_x - x[i])

                # Clamp velocity
                v[i] = np.clip(v[i], -max_v, max_v)

                # Update position
                x[i] = x[i] + v[i]
                x[i] = np.clip(x[i], -5, 5)

                # Evaluate
                fx[i] = func(x[i])
                calls_made += 1

                # Update personal best
                if fx[i] < pbest_f[i]:
                    pbest_f[i] = fx[i]
                    pbest_x[i] = x[i].copy()

                    # Update global best
                    if fx[i] < gbest_f:
                        gbest_f = fx[i]
                        gbest_x = x[i].copy()

        return gbest_x, gbest_f
