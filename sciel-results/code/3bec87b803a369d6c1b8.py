import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer for 5D Rastrigin:
    1. Structured coarse search (exploiting Rastrigin's periodic structure)
    2. Basin-jump ES with periodic jumps to escape local minima
    3. Fine-tuning with very small steps
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = [0]
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal best_x, best_f
            if calls[0] >= budget or time.time() > deadline:
                return None
            calls[0] += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Structured coarse search exploiting Rastrigin's periodic structure
        # Rastrigin has global minimum at 0 (before shift/rotation), so we
        # search around integer-valued candidates and the origin.
        candidates = []
        # Origin
        candidates.append(np.zeros(dim))
        # Integer-valued points in [-4,4] for each dimension
        for d in range(dim):
            for v in range(-4, 5):
                pt = np.zeros(dim)
                pt[d] = v
                candidates.append(pt)
        # Half-integer offsets
        for d in range(dim):
            for offset in [0.5, -0.5]:
                pt = np.zeros(dim)
                pt[d] = offset
                candidates.append(pt)
        # 2D slices: pairs of dimensions with integer values
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for v1 in range(-2, 3):
                    for v2 in range(-2, 3):
                        pt = np.zeros(dim)
                        pt[d1] = float(v1)
                        pt[d2] = float(v2)
                        candidates.append(pt)
        # Random uniform points
        for _ in range(30):
            candidates.append(rng.uniform(-5, 5, dim))

        # Shuffle candidates for diversity
        for i in range(len(candidates) - 1, 0, -1):
            j = rng.integers(0, i + 1)
            candidates[i], candidates[j] = candidates[j], candidates[i]

        # Evaluate up to 40 candidates
        for x in candidates:
            if calls[0] >= budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            call_func(x_clipped)

        # Phase 2: Basin-jump ES with periodic jumps
        # Use multiple starting points to handle the multi-modal Rastrigin
        starts = [best_x.copy()] if best_x is not None else [np.zeros(dim)]
        # Add a few diverse starting points
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))

        for start_x in starts:
            if calls[0] >= budget or time.time() > deadline:
                break

            x = start_x.copy()
            fx = best_f if (best_x is not None and np.array_equal(start_x, best_x)) else call_func(x)
            if fx is None:
                break

            sigma = 0.5
            no_improve = 0
            max_no_improve = 30
            jump_counter = 0

            while calls[0] < budget and time.time() < deadline:
                # Periodic jump: every 15 iterations, jump to a new random point
                # but keep the best found so far
                if jump_counter >= 15:
                    jump_counter = 0
                    # Jump to a new random point in a larger region
                    x = rng.uniform(-4, 4, dim)
                    fx = call_func(x)
                    if fx is None:
                        break
                    sigma = 0.5
                    no_improve = 0
                    continue

                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy is None:
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.15
                    no_improve = 0
                else:
                    sigma *= 0.92
                    no_improve += 1
                    jump_counter += 1

                if sigma < 0.005:
                    # Restart with larger sigma
                    sigma = 0.3
                    no_improve = 0
                    jump_counter = 0

                if no_improve >= max_no_improve:
                    break

        # Phase 3: Fine-tuning with very small steps near the best point
        if calls[0] < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            while calls[0] < budget and time.time() < deadline:
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy is None:
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.08
                else:
                    sigma *= 0.88

                if sigma < 0.001:
                    break

        return best_x, best_f
