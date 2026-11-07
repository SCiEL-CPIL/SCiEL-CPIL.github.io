"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derandomized Discrete PSO with ring topology for multimodal BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # PSO parameters
        swarm_size = 30
        v_max = 1.0
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5  # cognitive
        c2 = 1.5  # social

        # Initialize swarm
        positions = np.array([rng.uniform(-5, 5, dim) for _ in range(swarm_size)])
        velocities = np.zeros((swarm_size, dim))
        personal_bests = positions.copy()
        personal_best_f = np.array([np.inf] * swarm_size)

        # Ring topology: each particle interacts with 2 neighbors
        def get_neighbors(i):
            left = (i - 1) % swarm_size
            right = (i + 1) % swarm_size
            return [left, right]

        # Evaluate initial positions
        for i in range(swarm_size):
            if calls_made >= budget or time.time() > deadline:
                break
            fx = call_func(positions[i])
            personal_best_f[i] = fx
            personal_bests[i] = positions[i].copy()

        # Main PSO loop
        total_iters = budget - calls_made
        for it in range(total_iters):
            if calls_made >= budget or time.time() > deadline:
                break

            # Linearly decrease inertia weight
            progress = it / max(1, total_iters)
            w = w_start + (w_end - w_start) * progress

            # Update velocities and positions
            for i in range(swarm_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Get best from neighbors (ring topology)
                neighbors = get_neighbors(i)
                neighbor_best_idx = min(neighbors, key=lambda j: personal_best_f[j])

                # Derandomized PSO: use deterministic coefficients
                r1 = 0.5 + 0.5 * np.sin(2 * np.pi * (it / total_iters + i / swarm_size))
                r2 = 0.5 + 0.5 * np.cos(2 * np.pi * (it / total_iters + i / swarm_size))

                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_bests[i] - positions[i]) +
                                 c2 * r2 * (personal_bests[neighbor_best_idx] - positions[i]))

                # Clip velocity to v_max
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                # Evaluate
                fx = call_func(positions[i])
                if fx < personal_best_f[i]:
                    personal_best_f[i] = fx
                    personal_bests[i] = positions[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
