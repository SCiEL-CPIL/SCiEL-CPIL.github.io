"""Multi-start Nelder-Mead with quadratic basin-jumping for Lunacek bi-Rastrigin."""
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with quadratic model basin-jumping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, fx0, max_iter=50):
            """Nelder-Mead simplex method."""
            nonlocal calls_made
            if calls_made >= total_budget or time.time() > deadline:
                return x0, fx0

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fi = evaluate(xi)
                simplex.append(xi)
                fvals.append(fi)

            alpha, gamma, rho = 1.0, 2.0, 0.5
            max_no_improve = 20
            no_improve = 0

            for _ in range(max_iter):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                if no_improve >= max_no_improve:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    no_improve = 0
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    no_improve = 0
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = evaluate(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = evaluate(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                        fc = evaluate(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = evaluate(simplex[i])
                    no_improve += 1

                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            # Return best point in simplex
            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        def fit_quadratic(center, points, values):
            """Fit a full quadratic model and return the predicted minimizer."""
            m = len(points)
            if m < n + 1:
                return None

            X = np.ones((m, 1 + n + n * (n + 1) // 2))
            for k, (p, v) in enumerate(zip(points, values)):
                d = p - center
                col = 1
                for i in range(n):
                    X[k, col] = d[i]
                    col += 1
                for i in range(n):
                    for j in range(i, n):
                        X[k, col] = d[i] * d[j]
                        col += 1
            y = np.array(values)

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None

            g = coeffs[1:n + 1].copy()
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    hval = coeffs[idx]
                    idx += 1
                    if i == j:
                        H[i, j] = 2.0 * hval
                    else:
                        H[i, j] = hval
                        H[j, i] = hval

            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.any(eigvals <= 0):
                    return None
            except:
                return None

            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    return None

            model_center = coeffs[0]
            model_new = coeffs[0] + g @ dx + 0.5 * dx @ H @ dx
            if model_new >= model_center:
                return None

            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        def paraboloid_jump(center, f_center, h=2.0):
            """Fit paraboloid using 2n+1 points and return predicted minimizer."""
            nonlocal calls_made
            if calls_made + 2 * n + 1 > total_budget:
                return None
            if time.time() > deadline:
                return None

            points = [center.copy()]
            values = [f_center]

            for i in range(n):
                if calls_made >= total_budget or time.time() > deadline:
                    return None
                e_i = np.zeros(n)
                e_i[i] = 1.0
                x_plus = np.clip(center + h * e_i, -5, 5)
                x_minus = np.clip(center - h * e_i, -5, 5)
                fp = evaluate(x_plus)
                fm = evaluate(x_minus)
                points.append(x_plus)
                points.append(x_minus)
                values.append(fp)
                values.append(fm)

            return fit_quadratic(center, points, values)

        # Phase 1: Multi-start Nelder-Mead
        # Start from multiple initial points
        starts = [np.zeros(n)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, size=n))

        for start in starts:
            if calls_made >= total_budget or time.time() > deadline:
                break

            fs = evaluate(start)
            x_opt, f_opt = nelder_mead(start, fs, max_iter=40)

            # If improved, try basin jumping
            if f_opt < best_f and calls_made + 2 * n + 1 < total_budget and time.time() < deadline:
                jump_x = paraboloid_jump(x_opt, f_opt, h=1.5)
                if jump_x is not None:
                    jump_f = evaluate(jump_x)
                    if jump_f < f_opt:
                        x_opt, f_opt = nelder_mead(jump_x, jump_f, max_iter=30)

        # Phase 2: Final polishing with Nelder-Mead
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 10:
                best_x, best_f = nelder_mead(best_x, best_f, max_iter=30)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
