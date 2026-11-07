import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = None
        best_f = np.inf
        all_points = []

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64).copy()
            fx = float(func(x))
            calls += 1
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def local_refine(start_x, max_calls=50):
            nonlocal calls, best_x, best_f
            if calls >= budget - 15 or time_left() < 3.0:
                return
            x = np.asarray(start_x, dtype=np.float64).copy()
            fx = evaluate(x)

            # Adaptive coordinate descent
            step = 0.5
            no_improve = 0
            cd_calls = 0
            max_cd = max(20, max_calls // 2)

            for _ in range(max_cd):
                if calls >= budget - 15 or time_left() < 2.0:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget - 15 or time_left() < 2.0:
                        break
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        fx_new = evaluate(x_new)
                        cd_calls += 1
                        if fx_new < fx - 1e-10:
                            x = x_new
                            fx = fx_new
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-6 or no_improve > 5:
                        break
                else:
                    no_improve = 0
                    step = min(step * 1.3, 1.5)

            # Nelder-Mead polish
            if calls < budget - 10 and time_left() > 1.5:
                try:
                    from scipy.optimize import minimize
                    nm_calls = [0]
                    max_nm = max(10, max_calls - cd_calls)

                    def wrapped_func(x):
                        nonlocal calls, best_x, best_f
                        if nm_calls[0] >= max_nm or calls >= budget - 5 or time_left() < 1.0:
                            raise RuntimeError("budget exhausted")
                        x = np.asarray(x, dtype=np.float64)
                        fx = float(func(x))
                        calls += 1
                        nm_calls[0] += 1
                        all_points.append((fx, x.copy()))
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        return fx

                    minimize(
                        wrapped_func, x, method='Nelder-Mead',
                        options={'maxiter': max_nm, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
                except Exception:
                    pass

        # Phase 1: Global exploration with Sobol quasi-random sequence
        n_global = 100
        if dim <= 32:
            sobol = qmc.Sobol(d=dim, scramble=True)
            sobol_points = sobol.random(n_global)
            for i in range(n_global):
                if calls >= budget - 100 or time_left() < 5.0:
                    break
                x = -5.0 + 10.0 * sobol_points[i]
                evaluate(x)
        else:
            for i in range(n_global):
                if calls >= budget - 100 or time_left() < 5.0:
                    break
                x = np.random.uniform(-5, 5, dim)
                evaluate(x)

        # Add structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]
        for p in structured:
            if calls >= budget - 100 or time_left() < 5.0:
                break
            evaluate(p)

        # Phase 2: Select diverse top candidates and refine
        if all_points and calls < budget - 50 and time_left() > 10.0:
            all_points.sort(key=lambda t: t[0])
            candidates = []
            min_dist = 1.2
            for fx, x in all_points[:40]:
                if len(candidates) >= 6:
                    break
                if all(np.linalg.norm(x - cx) > min_dist for _, cx in candidates):
                    candidates.append((fx, x))

            if not candidates:
                candidates = [(fx, x) for fx, x in all_points[:4]]

            for fx, x in candidates:
                if calls >= budget - 30 or time_left() < 5.0:
                    break
                local_refine(x, max_calls=55)

        # Phase 3: Final polish
        if best_x is not None and calls < budget - 10 and time_left() > 3.0:
            local_refine(best_x, max_calls=30)

        return best_x, best_f
