"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Particle Swarm Optimization for f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        call_count = 0
        global_best_x = np.zeros(dim)
        global_best_fx = np.inf

        def record(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()

        rng = np.random.default_rng(42)

        # PSO parameters
        n_particles = 20
        w = 0.7       # inertia weight
        c1 = 1.49445  # cognitive coefficient
        c2 = 1.49445  # social coefficient
        v_max = 2.0   # velocity clamping

        # Initialize particles uniformly in the box
        positions = rng.uniform(-4, 4, size=(n_particles, dim))
        velocities = rng.uniform(-1, 1, size=(n_particles, dim))
        personal_best_x = positions.copy()
        personal_best_fx = np.full(n_particles, np.inf)

        # Evaluate initial positions
        for i in range(n_particles):
            if call_count >= budget:
                break
            fx = func(positions[i])
            call_count += 1
            personal_best_fx[i] = fx
            record(positions[i], fx)

        # Main PSO loop
        while call_count < budget:
            if time.time() > deadline - 5:
                break

            # Update velocities and positions
            for i in range(n_particles):
                if call_count >= budget:
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Velocity update
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_x[i] - positions[i]) +
                                 c2 * r2 * (global_best_x - positions[i]))

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Position update
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                fx = func(positions[i])
                call_count += 1

                # Update personal best
                if fx < personal_best_fx[i]:
                    personal_best_fx[i] = fx
                    personal_best_x[i] = positions[i].copy()
                    record(positions[i], fx)

            # Check for stagnation: if global best hasn't improved in a while,
            # reinitialize a fraction of particles
            if call_count % 50 == 0:
                # Reinitialize the worst half of particles
                sorted_idx = np.argsort(personal_best_fx)
                n_reinit = n_particles // 2
                for i in range(n_reinit):
                    idx = sorted_idx[n_particles - 1 - i]
                    if call_count >= budget:
                        break
                    positions[idx] = rng.uniform(-4, 4, size=dim)
                    velocities[idx] = rng.uniform(-1, 1, size=dim)
                    fx = func(positions[idx])
                    call_count += 1
                    personal_best_fx[idx] = fx
                    personal_best_x[idx] = positions[idx].copy()
                    record(positions[idx], fx)

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
