"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Particle Swarm Optimization (PSO) with inertia decay."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # PSO parameters
        n_particles = 20
        w_init = 0.9
        w_final = 0.4
        c1 = 1.49445  # cognitive
        c2 = 1.49445  # social

        # Initialize particles uniformly in [-5, 5]^dim
        positions = rng.uniform(-5, 5, (n_particles, dim))
        velocities = rng.uniform(-2, 2, (n_particles, dim))

        # Evaluate initial positions
        fitness = np.empty(n_particles)
        for i in range(n_particles):
            fitness[i] = func(positions[i])

        best_pos = positions.copy()
        best_fit = fitness.copy()
        global_best_idx = np.argmin(best_fit)
        global_best_pos = best_pos[global_best_idx].copy()
        global_best_fit = best_fit[global_best_idx]

        calls_made = n_particles

        # PSO main loop
        for iteration in range(budget - n_particles):
            # Check deadline
            if time.time() > deadline:
                break

            # Decay inertia weight
            progress = iteration / max(1, budget - n_particles - 1)
            w = w_init + (w_final - w_init) * progress

            for i in range(n_particles):
                # Check deadline more frequently
                if time.time() > deadline:
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Update velocity
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (best_pos[i] - positions[i])
                    + c2 * r2 * (global_best_pos - positions[i])
                )

                # Update position
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)

                # Evaluate
                new_fit = func(new_pos)
                calls_made += 1

                # Update personal best
                if new_fit < best_fit[i]:
                    best_fit[i] = new_fit
                    best_pos[i] = new_pos.copy()

                    # Update global best
                    if new_fit < global_best_fit:
                        global_best_fit = new_fit
                        global_best_pos = new_pos.copy()

                positions[i] = new_pos

            # Check deadline at end of each iteration
            if time.time() > deadline:
                break

        return global_best_pos, global_best_fit
# EVOLVE-BLOCK-END
