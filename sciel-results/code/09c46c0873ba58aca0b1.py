"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """PSO with adaptive inertia, restarts, and final polish for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        best_x_global = np.zeros(dim)
        best_f_global = float('inf')
        calls = 0

        def can_continue():
            return calls < budget and (time.time() - start_time) < max_time

        # PSO parameters
        n_particles = 20
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        w_init = 0.9  # Initial inertia weight
        w_min = 0.4  # Minimum inertia weight
        max_iter = 25  # Iterations per restart phase

        # Initialize particles
        particles = np.random.uniform(-4.5, 4.5, (n_particles, dim))
        velocities = np.random.uniform(-1.0, 1.0, (n_particles, dim))
        pbest_f = np.full(n_particles, float('inf'))
        pbest_x = np.zeros((n_particles, dim))
        gbest_f = float('inf')
        gbest_x = np.zeros(dim)

        # Evaluate initial particles
        for i in range(n_particles):
            if not can_continue():
                break
            x = np.clip(particles[i], -5, 5)
            fx = func(x)
            calls += 1
            pbest_f[i] = fx
            pbest_x[i] = x.copy()
            if fx < gbest_f:
                gbest_f = fx
                gbest_x = x.copy()
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()

        # Sort particles by fitness for diversity
        order = np.argsort(pbest_f)
        particles = particles[order]
        velocities = velocities[order]
        pbest_f = pbest_f[order]
        pbest_x = pbest_x[order]

        # Main PSO loop
        for it in range(max_iter):
            if not can_continue():
                break

            # Adapt inertia weight
            w = w_init - (w_init - w_min) * (it / max_iter)

            # Update velocities and positions
            for i in range(n_particles):
                if not can_continue():
                    break

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Velocity update
                velocities[i] = (w * velocities[i] +
                                c1 * r1 * (pbest_x[i] - particles[i]) +
                                c2 * r2 * (gbest_x - particles[i]))

                # Position update
                new_x = particles[i] + velocities[i]
                new_x = np.clip(new_x, -5, 5)

                # Evaluate
                fx = func(new_x)
                calls += 1

                # Update personal best
                if fx < pbest_f[i]:
                    pbest_f[i] = fx
                    pbest_x[i] = new_x.copy()
                    particles[i] = new_x.copy()

                    # Update global best
                    if fx < gbest_f:
                        gbest_f = fx
                        gbest_x = new_x.copy()
                    if fx < best_f_global:
                        best_f_global = fx
                        best_x_global = new_x.copy()
                else:
                    particles[i] = new_x.copy()

            # Check for stagnation: if no improvement in last few iterations, restart
            # Simple heuristic: if gbest hasn't improved significantly, add diversity
            if it > 0 and it % 5 == 0:
                # Reinitialize worst half of particles
                n_reinit = n_particles // 2
                for i in range(n_reinit, n_particles):
                    if not can_continue():
                        break
                    particles[i] = np.random.uniform(-4.5, 4.5, dim)
                    velocities[i] = np.random.uniform(-0.5, 0.5, dim)
                    # Keep pbest as is to maintain memory
                    # But evaluate new position
                    x = np.clip(particles[i], -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < pbest_f[i]:
                        pbest_f[i] = fx
                        pbest_x[i] = x.copy()
                        if fx < gbest_f:
                            gbest_f = fx
                            gbest_x = x.copy()
                        if fx < best_f_global:
                            best_f_global = fx
                            best_x_global = x.copy()

        # Final polish: fine-tuning around the best solution
        if calls < budget and time.time() - start_time < max_time:
            sigma = 0.05
            x = best_x_global.copy()
            fx = best_f_global
            for _ in range(100):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                direction = rng.standard_normal(dim)
                candidate = x + sigma * direction
                candidate = np.clip(candidate, -5, 5)
                fc = func(candidate)
                calls += 1
                if fc < fx:
                    x = candidate
                    fx = fc
                    sigma = min(sigma * 1.02, 0.3)
                    if fx < best_f_global:
                        best_f_global = fx
                        best_x_global = x.copy()
                else:
                    sigma = max(sigma * 0.98, 1e-5)

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
