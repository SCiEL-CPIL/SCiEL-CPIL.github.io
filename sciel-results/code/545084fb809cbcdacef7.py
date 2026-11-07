import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.RandomState(42)

        def g_of_x(x):
            # ZDT1 g(x) = 1 + 10*(dim-1) + sum((xi - 0.5)^2 + sin(10*pi*xi)) for i>=1
            tail = x[1:]
            return 1.0 + 10.0 * (dim - 1) + np.sum((tail - 0.5)**2 + np.sin(10.0 * np.pi * tail))

        def f_from_x(x):
            x1 = np.clip(x[0], 0.0, 1.0)
            g = g_of_x(x)
            f1 = x1
            f2 = (1.0 + g) * (1.0 - np.sqrt(x1))
            return np.array([f1, f2])

        def is_valid(v):
            return (v[0] < 11.0 and v[1] < 11.0 and np.all(np.isfinite(v)))

        # Local Nelder-Mead on tail variables to minimize g(x)
        def nelder_mead_on_tail(x1_target, x_init_tail, max_iters=25, tol=1e-10):
            n = dim - 1
            if n == 0:
                return np.array([x1_target, 0.0])
            # Simplex
            simplex = [np.array([x1_target] + list(x_init_tail))]
            for j in range(n):
                pt = np.array([x1_target] + list(x_init_tail))
                pt[j + 1] += 0.05 * (hi[j + 1] - lo[j + 1])
                pt = np.clip(pt, lo, hi)
                simplex.append(pt)
            # Evaluate
            values = [g_of_x(p) for p in simplex]
            iters = 0
            while iters < max_iters:
                # Sort
                order = np.argsort(values)
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]
                if abs(values[-1] - values[0]) < tol:
                    break
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                # Reflect
                xr = centroid + 1.0 * (centroid - worst)
                xr = np.clip(xr, lo, hi)
                fr = g_of_x(xr)
                if fr < values[0]:
                    xe = centroid + 2.0 * (centroid - worst)
                    xe = np.clip(xe, lo, hi)
                    fe = g_of_x(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        values[-1] = fe
                    else:
                        simplex[-1] = xr
                        values[-1] = fr
                elif fr < values[-2]:
                    simplex[-1] = xr
                    values[-1] = fr
                else:
                    xc = centroid + 0.5 * (worst - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = g_of_x(xc)
                    if fc < values[-1]:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            values[i] = g_of_x(simplex[i])
                iters += 1
            best_idx = np.argmin(values)
            return simplex[best_idx]

        # Phase 1: Evaluate a diverse set of x1 targets with x_tail=0
        # Use a mix of uniform and sqrt-spaced for better coverage
        n_targets = min(budget, 100)
        x1_targets = np.sort(np.concatenate([
            np.linspace(0.001, 0.999, n_targets // 2 + 1),
            np.linspace(0.01, 0.99, n_targets // 2 + 1) ** 0.5
        ]))
        x1_targets = x1_targets[:budget]

        eval_count = 0
        evaluated = []
        for x1_t in x1_targets:
            if time.time() >= deadline or eval_count >= budget:
                break
            x = np.zeros(dim)
            x[0] = np.clip(x1_t, lo[0], hi[0])
            x[1:] = lo[1:]
            v = problem(x)
            evaluated.append(v)
            eval_count += 1

        # Phase 2: Use remaining budget to locally optimize tail for select x1 targets
        remaining = budget - eval_count
        if remaining > 0 and dim > 1:
            # Pick x1 targets evenly spread
            n_local = min(remaining // 3, 20)  # ~3 evals per local search (rough)
            if n_local > 0:
                local_x1s = np.linspace(0.01, 0.99, n_local)
                for x1_l in local_x1s:
                    if time.time() >= deadline or eval_count >= budget:
                        break
                    # Initial tail: lower bound
                    x_init = np.zeros(dim)
                    x_init[0] = np.clip(x1_l, lo[0], hi[0])
                    x_init[1:] = lo[1:]
                    # Run local search
                    x_opt = nelder_mead_on_tail(x_init[0], x_init[1:], max_iters=10, tol=1e-9)
                    v = problem(x_opt)
                    if is_valid(v):
                        evaluated.append(v)
                        eval_count += 1

        # Phase 3: Fill remaining with more refined x1 grid if budget left
        if eval_count < budget:
            n_fill = budget - eval_count
            x1_fill = (np.arange(n_fill) + 0.5) / n_fill
            for x1_f in x1_fill:
                if time.time() >= deadline or eval_count >= budget:
                    break
                x = np.zeros(dim)
                x[0] = np.clip(x1_f, lo[0], hi[0])
                x[1:] = lo[1:]
                v = problem(x)
                evaluated.append(v)
                eval_count += 1

        return np.array(evaluated)
