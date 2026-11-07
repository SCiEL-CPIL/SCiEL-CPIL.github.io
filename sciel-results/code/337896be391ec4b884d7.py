import numpy as np
import time

class Optimizer:
    """Hooke-Jeeves pattern search for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

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

        def hooke_jees(start_x, max_iter=500):
            nonlocal eval_count
            if eval_count >= budget:
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget or time.time() > deadline:
                return

            step_sizes = np.ones(d) * 1.0
            min_step = 1e-7
            prev_x = x.copy()

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget - 3 or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget or time.time() > deadline:
                        break

                    if f_plus < f:
                        # Line search: try to go further in this direction
                        step = step_sizes[i]
                        for _ls in range(3):
                            if eval_count >= budget - 1 or time.time() > deadline:
                                break
                            step *= 2.0
                            x_trial = x.copy()
                            x_trial[i] += step
                            f_trial = eval_func(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                step_sizes[i] = step
                            else:
                                break
                        moved = True
                        continue
                    else:
                        step_sizes[i] *= 0.5

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget or time.time() > deadline:
                        break

                    if f_minus < f:
                        # Line search: try to go further in this direction
                        step = step_sizes[i]
                        for _ls in range(3):
                            if eval_count >= budget - 1 or time.time() > deadline:
                                break
                            step *= 2.0
                            x_trial = x.copy()
                            x_trial[i] -= step
                            f_trial = eval_func(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                step_sizes[i] = step
                            else:
                                break
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                # Pattern move: try to continue in the direction of movement
                if moved and eval_count < budget - 5 and time.time() < deadline:
                    direction = x - prev_x
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                        for _pm in range(2):
                            if eval_count >= budget - 1 or time.time() > deadline:
                                break
                            x_trial = x + direction * 1.0
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = eval_func(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                            else:
                                break

                prev_x = x.copy()

                if np.all(step_sizes < min_step):
                    break

        # Start from origin
        hooke_jees(np.zeros(d))

        # Strategic restarts
        rng = np.random.default_rng(42)
        restarts = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            rng.uniform(-3, 3, d),
            rng.uniform(-5, 5, d),
            np.array([0.0, -2.0, 0.0, 0.0, 0.0]),
            rng.uniform(-2, 2, d),
        ]

        for r in restarts:
            if eval_count >= budget - 20 or time.time() > deadline:
                break
            hooke_jees(r)

        # Final fine-tuning
        if eval_count < budget - 10 and time.time() < deadline:
            hooke_jees(best_x.copy())

        return best_x, best_f
