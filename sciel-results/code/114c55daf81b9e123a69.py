"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with window-based success-rate adaptation, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 0: Multi-start initialization to find both basins of Lunacek
        # Lunacek bi-Rastrigin has a global minimum near 0 and a local minimum at offset
        init_points = [
            np.zeros(n),
            rng.uniform(-1, 1, n),
            rng.uniform(-2, 2, n),
            np.ones(n) * 2.5,
            np.ones(n) * -2.5,
            rng.uniform(-4, 4, n),
        ]
        for xp in init_points:
            if calls_made >= total_budget:
                break
            safe_func(xp)

        # Phase 1: Population-based PSO to explore both basins
        # PSO with inertia, cognitive and social components
        pop_size = 12
        c1 = 1.5  # cognitive
        c2 = 1.5  # social
        w_start = 0.9
        w_end = 0.4
        max_vel = 2.0

        # Initialize population around diverse starting points
        # Include the best found point and spread others
        particles = []
        velocities = []
        pbests = []
        pbest_fs = []

        # First particle at best
        particles.append(best_x.copy())
        pbests.append(best_x.copy())
        pbest_fs.append(best_f)
        velocities.append(np.zeros(n))

        # Spread remaining particles
        spread_centers = [
            np.zeros(n),
            rng.uniform(-3, 3, n),
            rng.uniform(-4, 4, n),
            np.ones(n) * 3.0,
            np.ones(n) * -3.0,
            rng.uniform(-2, 2, n),
        ]
        for i in range(1, pop_size):
            center = spread_centers[(i - 1) % len(spread_centers)]
            p = center + 0.5 * rng.standard_normal(n)
            p = np.clip(p, -5, 5)
            particles.append(p)
            velocities.append(np.zeros(n))
            if calls_made < total_budget:
                fp = safe_func(p)
            else:
                fp = best_f
            pbests.append(p.copy())
            pbest_fs.append(fp)

        # PSO main loop
        pso_budget = int(total_budget * 0.55)  # use 55% of budget for PSO
        pso_calls = 0
        max_pso_iters = pso_budget // pop_size

        for it in range(max_pso_iters):
            if calls_made >= total_budget or time.time() >= deadline - 15:
                break
            # Linear inertia decay
            w = w_start + (w_end - w_start) * (it / max(1, max_pso_iters))

            for i in range(pop_size):
                if calls_made >= total_budget or time.time() >= deadline - 15:
                    break

                r1 = rng.random(n)
                r2 = rng.random(n)
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbests[i] - particles[i])
                                 + c2 * r2 * (best_x - particles[i]))
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                new_p = particles[i] + velocities[i]
                new_p = np.clip(new_p, -5, 5)
                particles[i] = new_p

                fp = safe_func(new_p)

                if fp < pbest_fs[i]:
                    pbest_fs[i] = fp
                    pbests[i] = new_p.copy()

        # Phase 2: (1+1)-ES local refinement from best point
        if time.time() < deadline - 5 and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            stagnation_count = 0
            max_stagnation = 25
            window = []
            window_size = 12

            remaining = total_budget - calls_made
            for step in range(remaining):
                if time.time() > deadline - 5:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    stagnation_count = 0
                    window.append(1)
                else:
                    stagnation_count += 1
                    window.append(0)

                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.35:
                        sigma *= 1.08
                    elif success_rate < 0.1:
                        sigma *= 0.92
                    window = window[1:]

                if stagnation_count >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
