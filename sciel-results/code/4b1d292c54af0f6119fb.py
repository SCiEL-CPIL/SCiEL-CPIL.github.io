import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start, adaptive restart, and polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        calls = 0

        def eval_func(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            return func(x)

        # Phase 1: Initial exploration - evaluate several random points
        n_init = min(10, self.budget // 10)
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = eval_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Nelder-Mead optimization
        # Build initial simplex around best_x
        simplex = [best_x.copy()]
        for i in range(n):
            xi = best_x.copy()
            step = 0.5 if abs(best_x[i]) > 0.1 else 0.1
            xi[i] += step
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            f = eval_func(xi)
            # Store f values
            if len(simplex) == i + 2:
                simplex[i + 1] = (xi, f)
            else:
                simplex[i + 1] = xi

        # Proper simplex construction
        simplex = []
        simplex.append((best_x.copy(), best_f))
        for i in range(n):
            xi = best_x.copy()
            step = 0.5 if abs(best_x[i]) > 0.1 else 0.1
            xi[i] += step
            xi = np.clip(xi, -5.0, 5.0)
            f = eval_func(xi)
            simplex.append((xi, f))

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = self.budget - calls - 5  # leave some for polish
        stall_count = 0
        stall_threshold = 20
        best_x_nm = best_x.copy()
        best_f_nm = best_f

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= self.budget - 5:
                break

            # Sort simplex by function value
            simplex.sort(key=lambda sf: sf[1])
            f_worst = simplex[-1][1]
            f_best = simplex[0][1]

            # Update best
            if f_best < best_f_nm:
                best_f_nm = f_best
                best_x_nm = simplex[0][0].copy()

            # Check convergence
            if abs(f_worst - f_best) < 1e-12:
                break

            # Centroid of all but worst point
            centroid = np.mean([pt for pt, f in simplex[:-1]], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1][0])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = eval_func(x_r)

            if f_r < f_best:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1][0])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = eval_func(x_e)
                if f_e < f_r:
                    simplex[-1] = (x_e, f_e)
                else:
                    simplex[-1] = (x_r, f_r)
                stall_count = 0
            elif f_r < simplex[-2][1]:
                simplex[-1] = (x_r, f_r)
                stall_count = 0
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = eval_func(x_c)
                if f_c < min(f_r, f_worst):
                    simplex[-1] = (x_c, f_c)
                    stall_count = 0
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        x_s = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                        x_s = np.clip(x_s, -5.0, 5.0)
                        f_s = eval_func(x_s)
                        simplex[i] = (x_s, f_s)
                    stall_count += 1

            # Restart if stalled
            if stall_count >= stall_threshold:
                stall_count = 0
                # Perturb best point and restart
                x_new = best_x_nm + 0.3 * rng.standard_normal(n)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = eval_func(x_new)
                if f_new < best_f_nm:
                    best_f_nm = f_new
                    best_x_nm = x_new.copy()
                # Rebuild simplex
                simplex = [(best_x_nm.copy(), best_f_nm)]
                for i in range(n):
                    xi = best_x_nm.copy()
                    step = 0.2 if abs(best_x_nm[i]) > 0.1 else 0.05
                    xi[i] += step
                    xi = np.clip(xi, -5.0, 5.0)
                    f = eval_func(xi)
                    simplex.append((xi, f))

        # Phase 3: Polish phase - tight tolerance local search
        polish_budget = self.budget - calls
        if polish_budget > 10 and time.time() < deadline:
            # Use a very fine Nelder-Mead or coordinate descent
            # Let's do a fine NM with small initial steps
            x_polish = best_x_nm.copy()
            f_polish = best_f_nm
            step_size = 0.01
            for _ in range(min(polish_budget - 2, 100)):
                if time.time() > deadline or calls >= self.budget:
                    break
                improved = False
                for i in range(n):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    for direction in [1, -1]:
                        x_try = x_polish.copy()
                        x_try[i] += direction * step_size
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = eval_func(x_try)
                        if f_try < f_polish:
                            x_polish = x_try
                            f_polish = f_try
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

            if f_polish < best_f_nm:
                best_x_nm = x_polish
                best_f_nm = f_polish

        return best_x_nm, best_f_nm
