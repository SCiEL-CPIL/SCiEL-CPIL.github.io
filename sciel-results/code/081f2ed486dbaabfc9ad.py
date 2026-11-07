import numpy as np
import time

class Optimizer:
    """Hooke-Jeeves pattern search for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def hooke_jees(start_x):
            nonlocal eval_count
            if eval_count >= budget:
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            step_sizes = np.ones(d) * 0.5
            min_step = 1e-6
            max_iter = 1000

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= 2.0
                        moved = True
                        # Keep exploring this direction
                        continue
                    else:
                        step_sizes[i] *= 0.5

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget:
                        break

                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        step_sizes[i] *= 2.0
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                if np.all(step_sizes < min_step):
                    break

        # Start from origin
        hooke_jees(np.zeros(d))

        # If budget remains, try a few random restarts
        if eval_count < budget - 20:
            rng = np.random.default_rng(12345)
            for _ in range(10):
                if eval_count >= budget - 10:
                    break
                x_rand = rng.uniform(-5, 5, d)
                hooke_jees(x_rand)

        return best_x, best_f
