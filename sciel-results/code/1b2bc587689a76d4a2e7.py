import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill

        calls = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Exploration to identify the landscape structure
        # Sample diverse points to understand the shift and rotation
        n_init = 30
        candidates = []

        # Always try the origin
        eval_func(np.zeros(dim))

        # Sample a grid of points to better understand the periodic structure
        # Rastrigin has minima at multiples of 2*pi in the rotated frame
        # Try points at various offsets to detect the shift
        for _ in range(n_init - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Generate candidate minima using periodicity
        # Rastrigin's local minima in the standard frame are at integer points
        # In the rotated+shifted frame, they're at shifted integer points
        # Generate a lattice of candidate points based on the best initial estimate
        # The minima should be near integer coordinates in the rotated frame

        # Use the best initial candidate to estimate where the global minimum might be
        # Generate candidates at integer-like positions around the best found point
        lattice_candidates = []

        # Add the top initial candidates
        for f, x in candidates[:5]:
            lattice_candidates.append((f, x))

        # Generate additional candidates based on periodicity
        # Try points at integer coordinates (in the original frame, since we don't know the rotation)
        # This is a heuristic: the minima are likely near integer points after accounting for shift
        for i in range(-2, 3):
            for j in range(-2, 3):
                if dim >= 2:
                    x_test = np.zeros(dim)
                    x_test[0] = float(i)
                    x_test[1] = float(j)
                    if calls < budget and time.time() < deadline:
                        f_test = eval_func(x_test)
                        lattice_candidates.append((f_test, x_test.copy()))

        # Also try points at half-integer positions
        for i in range(-2, 3):
            for j in range(-2, 3):
                if dim >= 2:
                    x_test = np.zeros(dim)
                    x_test[0] = float(i) + 0.5
                    x_test[1] = float(j) + 0.5
                    if calls < budget and time.time() < deadline:
                        f_test = eval_func(x_test)
                        lattice_candidates.append((f_test, x_test.copy()))

        # Sort all lattice candidates
        lattice_candidates.sort(key=lambda t: t[0])

        # Phase 3: Multi-start (1+1)-ES refinement on top candidates
        top_k = 6
        max_local_budget = 70
        min_local_budget = 15

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if i >= len(lattice_candidates):
                break

            f_start, x_start = lattice_candidates[i]

            # Skip if this candidate is much worse than current global best
            if i > 0 and f_start > best_f + 2.0:
                continue

            x = x_start.copy()
            f_x = f_start

            # (1+1)-ES Parameters
            sigma = 0.4  # Initial step size
            window = 10
            successes = 0
            stagnation = 0
            max_stagnation = 15
            local_calls = 0
            current_budget = min_local_budget

            while calls < budget and time.time() < deadline and local_calls < max_local_budget:
                # Early exit if this candidate is clearly worse
                if local_calls >= current_budget and f_x > best_f + 0.5:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                local_calls += 1

                if f_y < f_x:
                    x = y
                    f_x = f_y
                    successes += 1
                    stagnation = 0
                    # Increase budget if improving
                    if local_calls % 8 == 0:
                        current_budget = min(max_local_budget, current_budget + 8)
                else:
                    stagnation += 1

                # Adapt sigma based on success rate
                if successes > 0 and successes % window == 0:
                    sr = successes / window
                    if sr > 0.5:
                        sigma *= 1.25
                    elif sr < 0.3:
                        sigma *= 0.75
                    sigma = np.clip(sigma, 0.02, 3.0)
                    successes = 0

                # Stagnation handling: reset sigma to escape local minimum
                if stagnation > max_stagnation:
                    sigma = max(sigma, 0.4)
                    stagnation = 0

        # Phase 4: Final Polish
        # Fine-grained search around the best point found
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.03
            remaining = budget - calls
            max_polish_steps = min(remaining, 60)

            for _ in range(max_polish_steps):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    sigma *= 0.92
                else:
                    sigma *= 0.96
                sigma = max(sigma, 0.001)

        return best_x, best_f
