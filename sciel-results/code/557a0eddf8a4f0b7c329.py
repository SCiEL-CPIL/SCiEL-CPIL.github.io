"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derandomized Discrete Particle Swarm Optimization (ddPSO) with ring topology."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget
        N = 20  # swarm size
        max_vel = 1.0

        # Initialize particles
        particles = rng.uniform(-5, 5, (N, dim))
        personal_bests = particles.copy()
        personal_best_vals = np.array([func(p) for p in particles])

        global_best_idx = np.argmin(personal_best_vals)
        global_best = personal_bests[global_best_idx].copy()
        global_best_val = personal_best_vals[global_best_idx]

        # Ring topology: particle i looks at particle (i+1) % N
        velocities = np.zeros((N, dim))
        w_start, w_end = 0.9, 0.4
        c1 = c2 = 1.5  # cognitive and social coefficients

        call_count = N  # initial evaluations

        for it in range(budget - N):
            if time.time() > deadline:
                break

            # Dynamic inertia weight
            w = w_start + (w_end - w_start) * (it / max(1, budget - N - 1))

            for i in range(N):
                if time.time() > deadline:
                    break

                # Ring topology neighbor
                j = (i + 1) % N

                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (personal_bests[i] - particles[i])
                                 + c2 * r2 * (personal_bests[j] - particles[i]))

                # Clip velocity
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                # Update position
                new_pos = particles[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)

                # Evaluate
                new_val = func(new_pos)
                call_count += 1

                particles[i] = new_pos

                # Update personal best
                if new_val < personal_best_vals[i]:
                    personal_best_vals[i] = new_val
                    personal_bests[i] = new_pos.copy()

                    # Update global best
                    if new_val < global_best_val:
                        global_best_val = new_val
                        global_best = new_pos.copy()

        return global_best, global_best_val
# EVOLVE-BLOCK-END
