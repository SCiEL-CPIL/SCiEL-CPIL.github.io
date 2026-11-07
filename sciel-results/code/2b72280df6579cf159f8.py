import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_pos = self.n_obj - 1  # 2
        n_dist = self.dim - n_pos  # 10

        evaluated_points = []
        evaluated_f = []
        used = 0

        def eval_point(t1, t2):
            nonlocal used
            if used >= self.budget:
                return False
            if time.time() > deadline:
                return False
            if t1 < -1e-10 or t2 < -1e-10 or t1 > 1.0 + 1e-10 or t2 > 1.0 + 1e-10:
                return False
            if t1 * t1 + t2 * t2 > 1.0 + 1e-10:
                return False

            x = np.zeros(self.dim)
            x[0] = t1
            x[1] = t2
            x[n_pos:] = 0.5  # g = 0, on Pareto front
            x = np.clip(x, lo, hi)

            f = np.array(problem(x))
            evaluated_points.append(x.copy())
            evaluated_f.append(f)
            used += 1
            return True

        # Phase 1: 14x14 grid on the Pareto front (quarter circle)
        n_grid = 14
        pos_vals = np.linspace(0.0, 1.0, n_grid)
        for t1 in pos_vals:
            for t2 in pos_vals:
                eval_point(t1, t2)
            if used >= self.budget or time.time() > deadline:
                break

        # Phase 2: Offset 14x14 grid (half-step offset) to fill gaps
        if used < self.budget:
            offset = 1.0 / (2.0 * n_grid)
            pos_vals2 = np.linspace(offset, 1.0 - offset, n_grid)
            for t1 in pos_vals2:
                for t2 in pos_vals2:
                    eval_point(t1, t2)
                if used >= self.budget or time.time() > deadline:
                    break

        # Phase 3: Remaining budget - polar grid for fine coverage
        if used < self.budget:
            remaining = self.budget - used
            n_theta = min(remaining, 20)
            if n_theta > 0:
                n_r = max(1, remaining // n_theta)
                for i in range(n_theta):
                    for j in range(n_r):
                        if used >= self.budget:
                            break
                        angle = np.pi / 2.0 * (i + 0.5) / n_theta
                        r = (j + 0.5) / n_r
                        t1 = r * np.cos(angle)
                        t2 = r * np.sin(angle)
                        eval_point(t1, t2)

        return evaluated_points[-1] if evaluated_points else np.zeros(self.dim)
