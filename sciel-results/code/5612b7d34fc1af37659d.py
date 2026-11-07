import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 280.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Initial Global Search using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2)
            return points

        global_calls = 20
        halton_points = halton_sequence(global_calls, dim)
        halton_points = halton_points * 10 - 5

        for x in halton_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        remaining = total_budget - calls_made

        # Phase 2: CMA-ES-lite with coordinate-wise adaptive step sizes
        # Each dimension has an independent step size that adapts based on success rate
        step_sizes = np.ones(dim) * 0.5
        success_window = 20  # number of recent trials to track
        successes = np.zeros(dim, dtype=int)
        trials = np.zeros(dim, dtype=int)
        trial_count = 0

        # Track recent successes for each dimension
        recent_successes = {i: [] for i in range(dim)}

        polish_interval = 15  # every N iterations, do a gradient polish
        iteration = 0

        while remaining > 2 * dim + 5 and time.time() < deadline:
            iteration += 1

            # Periodic gradient polish using finite differences
            if iteration % polish_interval == 0 and remaining > 3 * dim:
                # Estimate gradient via central differences
                eps = 1e-4
                grad = np.zeros(dim)
                for i in range(dim):
                    if time.time() > deadline or calls_made >= total_budget:
                        break
                    x_plus = best_x.copy()
                    x_plus[i] += eps
                    x_minus = best_x.copy()
                    x_minus[i] -= eps
                    try:
                        fp = wrapper(x_plus)
                        fm = wrapper(x_minus)
                        grad[i] = (fp - fm) / (2 * eps)
                    except TimeoutError:
                        break
                    remaining = total_budget - calls_made

                if time.time() > deadline or calls_made >= total_budget:
                    break

                # Gradient descent step with adaptive learning rate
                lr = 0.01
                for _ in range(min(5, max(1, (total_budget - calls_made) // (2 * dim)))):
                    if time.time() > deadline or calls_made >= total_budget:
                        break
                    step = -lr * grad
                    # Limit step size per dimension
                    step = np.clip(step, -0.1, 0.1)
                    candidate = np.clip(best_x + step, -5, 5)
                    try:
                        fc = wrapper(candidate)
                    except TimeoutError:
                        break
                    if fc < best_f:
                        best_f = fc
                        best_x = candidate.copy()
                        # Re-estimate gradient
                        grad = np.zeros(dim)
                        for i in range(dim):
                            if time.time() > deadline or calls_made >= total_budget:
                                break
                            x_plus = best_x.copy()
                            x_plus[i] += eps
                            x_minus = best_x.copy()
                            x_minus[i] -= eps
                            try:
                                fp = wrapper(x_plus)
                                fm = wrapper(x_minus)
                                grad[i] = (fp - fm) / (2 * eps)
                            except TimeoutError:
                                break
                            remaining = total_budget - calls_made
                    remaining = total_budget - calls_made

                if time.time() > deadline or calls_made >= total_budget:
                    break

            # CMA-ES-lite: propose candidate with adaptive per-dimension steps
            if time.time() > deadline or calls_made >= total_budget:
                break

            # Generate random direction
            direction = rng.normal(0, 1, dim)
            candidate = best_x + direction * step_sizes
            candidate = np.clip(candidate, -5, 5)

            try:
                fc = wrapper(candidate)
            except TimeoutError:
                break

            remaining = total_budget - calls_made

            # Update per-dimension success statistics
            for i in range(dim):
                success = fc < best_f
                recent_successes[i].append(1 if success else 0)
                if len(recent_successes[i]) > success_window:
                    recent_successes[i].pop(0)
                # Update step size based on success rate
                if len(recent_successes[i]) >= 5:
                    rate = np.mean(recent_successes[i])
                    if rate > 0.5:
                        step_sizes[i] *= 1.15  # expand
                    elif rate < 0.3:
                        step_sizes[i] *= 0.85  # shrink
                    # Clamp step sizes
                    step_sizes[i] = np.clip(step_sizes[i], 0.01, 3.0)

            if fc < best_f:
                best_f = fc
                best_x = candidate.copy()

            # Occasional random restart to escape local minima
            if iteration % 30 == 0 and remaining > 10 * dim:
                # Perturb significantly in a random direction
                restart_step = np.random.uniform(0.5, 2.0)
                restart_dir = rng.normal(0, 1, dim)
                restart_dir /= (np.linalg.norm(restart_dir) + 1e-10)
                candidate = np.clip(best_x + restart_step * restart_dir, -5, 5)
                try:
                    fc = wrapper(candidate)
                except TimeoutError:
                    break
                if fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()
                remaining = total_budget - calls_made

        return best_x, best_f
