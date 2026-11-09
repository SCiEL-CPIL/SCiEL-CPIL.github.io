import numpy as np
import time


class Optimizer:
    """Multi-start coordinate descent with golden-section line search for Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def time_ok():
            return time.time() < deadline and calls < budget

        def eval_point(x):
            nonlocal calls, best_f, best_x
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        def golden_section_1d(x, d, lo, hi, max_evals=10):
            if not time_ok():
                return x, best_f
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            c = b - gr * (b - a)
            dd = a + gr * (b - a)
            fc = eval_point(x + c * d)
            if not time_ok():
                return x + c * d, fc
            fd = eval_point(x + dd * d)
            if not time_ok():
                return x + dd * d, fd
            for _ in range(max(max_evals - 2, 0)):
                if not time_ok():
                    break
                if fc < fd:
                    b = dd
                    dd = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = eval_point(x + c * d)
                else:
                    a = c
                    c = dd
                    fc = fd
                    dd = a + gr * (b - a)
                    fd = eval_point(x + dd * d)
            if fc <= fd:
                return x + c * d, fc
            else:
                return x + dd * d, fd

        rng = np.random.default_rng(42)

        # Phase 1: Random initialization
        for _ in range(25):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            eval_point(x)

        if best_f == float('inf'):
            eval_point(np.zeros(dim))

        # Phase 2: Coordinate descent with golden-section line search
        for restart in range(3):
            if not time_ok():
                break
            if restart == 0:
                x = best_x.copy()
                x_f = best_f
            else:
                x = best_x + rng.standard_normal(dim) * 0.3
                x = np.clip(x, -5, 5)
                x_f = eval_point(x)

            for cyc in range(6):
                if not time_ok():
                    break
                cyc_improved = False
                for i in range(dim):
                    if not time_ok():
                        break
                    d = np.zeros(dim)
                    d[i] = 1.0
                    new_x, new_f = golden_section_1d(x, d, -8.0, 8.0, max_evals=10)
                    if new_f < x_f:
                        x = new_x
                        x_f = new_f
                        cyc_improved = True
                if not cyc_improved:
                    break

            if restart > 0 and x_f >= best_f:
                break

        return best_x, best_f
