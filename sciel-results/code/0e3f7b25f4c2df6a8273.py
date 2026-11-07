"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization with inertia decay and constriction factor."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s out of 300s limit for safety

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        n_particles = 20

        # Initialize particles uniformly in [-5, 5]^dim
        positions = rng.uniform(-5, 5, (n_particles, dim))
        velocities = rng.uniform(-1, 1, (n_particles, dim))

        # Evaluate initial population
        fitness = np.empty(n_particles)
        for i in range(n_particles):
            x = np.clip(positions[i], -5, 5)
            fitness[i] = func(x)

        # Track personal and global bests
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fit = personal_best_fit[global_best_idx]

        # PSO parameters
        w_initial = 0.9
        w_final = 0.4
        c1 = 1.49445  # cognitive
        c2 = 1.49445  # social

        calls_used = n_particles

        # Main PSO loop
        for it in range(budget - n_particles):
            # Check time deadline
            if time.time() > deadline:
                break

            # Update inertia weight linearly
            progress = it / max(1, budget - n_particles)
            w = w_initial + (w_final - w_initial) * progress

            # Constriction factor
            A = w + c1 + c2
            if A > 2:
                phi = c1 + c2
                chi = 2.0 / (2.0 - phi - np.sqrt(max(0, phi * phi - 2 * phi)))
            else:
                chi = 1.0

            for i in range(n_particles):
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best_pos[i] - positions[i])
                    + c2 * r2 * (global_best_pos - positions[i])
                )
                velocities[i] *= chi

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                x = positions[i]
                fx = func(x)
                calls_used += 1

                # Update personal best
                if fx < personal_best_fit[i]:
                    personal_best_fit[i] = fx
                    personal_best_pos[i] = x.copy()
                    # Update global best
                    if fx < global_best_fit:
                        global_best_fit = fx
                        global_best_pos = x.copy()

                # Check time deadline mid-iteration
                if time.time() > deadline:
                    break

            if time.time() > deadline:
                break

        # Final polish: local search around global best using remaining budget
        remaining = budget - calls_used
        if remaining > 0 and time.time() < deadline:
            x = global_best_pos.copy()
            fx = global_best_fit
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return global_best_pos, global_best_fit
# EVOLVE-BLOCK-END
