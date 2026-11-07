import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def golden_section_1d(x, d, lo, hi, f_lo, f_hi, max_iter=30):
            """1D golden-section search along direction d from x."""
            nonlocal calls
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a = lo
            b = hi
            fa = f_lo
            fb = f_hi

            for _ in range(max_iter):
                if calls >= budget - 5 or time.time() > deadline - 5:
                    break
                c = b - gr * (b - a)
                d_ = a + gr * (b - a)
                fc = try_eval(x + c * d)
                fd = try_eval(x + d_ * d)
                if fc is None or fd is None:
                    break
                if fc > fd:
                    a = c
                    fa = fc
                    b = d_
                    fb = fd
                else:
                    b = d_
                    fb = fd
                    a = c
                    fa = fc
            # Return the better of the two interior points
            if fa < fb:
                return x + a * d, fa
            else:
                return x + b * d, fb

        def powell_method(x0, f0, max_iter=15):
            """Powell's conjugate direction method."""
            nonlocal calls, best_x, best_f
            if calls >= budget - 20 or time.time() > deadline - 10:
                return x0, f0

            x = x0.copy()
            f = f0
            directions = np.eye(n)

            for iteration in range(max_iter):
                if calls >= budget - 10 or time.time() > deadline - 5:
                    break

                x_min = x.copy()
                f_min = f
                d_min = directions[0]

                for i in range(n):
                    if calls >= budget - 10 or time.time() > deadline - 5:
                        break
                    d = directions[i]
                    # Find the minimum along direction d
                    # First, determine the bracket
                    step = 0.1
                    x_plus = x + step * d
                    f_plus = try_eval(x_plus)
                    if f_plus is None:
                        break
                    x_minus = x - step * d
                    f_minus = try_eval(x_minus)
                    if f_minus is None:
                        break

                    if f_plus < f:
                        # Minimum is in the positive direction
                        lo, hi = 0.0, step
                        f_lo, f_hi = f, f_plus
                        x_new, f_new = golden_section_1d(x, d, lo, hi, f_lo, f_hi)
                    elif f_minus < f:
                        # Minimum is in the negative direction
                        lo, hi = -step, 0.0
                        f_lo, f_hi = f_minus, f
                        x_new, f_new = golden_section_1d(x, d, lo, hi, f_lo, f_hi)
                    else:
                        # Try a larger step
                        step *= 2
                        x_plus = x + step * d
                        f_plus = try_eval(x_plus)
                        if f_plus is None:
                            break
                        x_minus = x - step * d
                        f_minus = try_eval(x_minus)
                        if f_minus is None:
                            break
                        if f_plus < f:
                            lo, hi = 0.0, step
                            f_lo, f_hi = f, f_plus
                            x_new, f_new = golden_section_1d(x, d, lo, hi, f_lo, f_hi)
                        elif f_minus < f:
                            lo, hi = -step, 0.0
                            f_lo, f_hi = f_minus, f
                            x_new, f_new = golden_section_1d(x, d, lo, hi, f_lo, f_hi)
                        else:
                            continue

                    if f_new < f:
                        f = f_new
                        x = x_new
                        if f < f_min:
                            f_min = f
                            x_min = x.copy()
                            d_min = d

                # Update directions: replace the direction that gave the smallest improvement
                if np.linalg.norm(x - x0) > 1e-8:
                    d_new = x - x0
                    directions[n - 1] = d_min
                    # Rotate directions
                    for i in range(n - 1, 0, -1):
                        directions[i] = directions[i - 1]
                    directions[0] = d_new

                # Check for convergence
                if np.linalg.norm(x - x0) < 1e-6 and abs(f - f0) < 1e-8:
                    break

            if f < best_f:
                best_x = x.copy()
                best_f = f
            return x, f

        def nelder_mead(x0, f0, max_iter=20):
            """Nelder-Mead simplex method."""
            nonlocal calls, best_x, best_f
            if calls >= budget - 30 or time.time() > deadline - 10:
                return x0, f0

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.25
                simplex.append(xi)
                f = try_eval(xi)
                if f is None:
                    break
                fvals.append(f)
            else:
                # Only proceed if we have a full simplex
                for iteration in range(max_iter):
                    if calls >= budget - 10 or time.time() > deadline - 5:
                        break

                    # Sort simplex by function value
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    fr = try_eval(xr)
                    if fr is None:
                        break

                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        fe = try_eval(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = try_eval(xc)
                        if fc is None:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f = try_eval(simplex[i])
                                if f is None:
                                    return x0, f0
                                fvals[i] = f

                    # Update best
                    if fvals[0] < best_f:
                        best_f = fvals[0]
                        best_x = simplex[0].copy()

            # Find the best point in the simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 1: Global sampling with Halton sequence
        def halton(index, dim):
            """Generate Halton sequence point."""
            point = []
            for j in range(dim):
                prime = [2, 3, 5, 7, 11][j]
                f = 1.0
                result = 0.0
                i = index
                while i > 0:
                    f /= prime
                    result += f * (i % prime)
                    i //= prime
                point.append(result * 10.0 - 5.0)
            return np.array(point)

        # Sample 60 points with Halton sequence for global coverage
        n_global = 60
        for i in range(1, n_global + 1):
            if calls >= budget - 40 or time.time() > deadline - 20:
                break
            x = halton(i, n)
            try_eval(x)

        # Add 10 random points for diversity
        for _ in range(10):
            if calls >= budget - 40 or time.time() > deadline - 20:
                break
            x = rng.uniform(-5, 5, n)
            try_eval(x)

        # Phase 2: Local refinement with Powell's method from best point
        if best_f < float('inf') and calls < budget - 20:
            remaining = budget - calls
            if remaining >= 30:
                x_best, f_best = powell_method(best_x, best_f, max_iter=12)
                if f_best < best_f:
                    best_x = x_best
                    best_f = f_best

        # Phase 3: If budget remains, try Nelder-Mead from best point
        if calls < budget - 20 and time.time() < deadline - 10:
            remaining = budget - calls
            if remaining >= 30:
                x_nm, f_nm = nelder_mead(best_x, best_f, max_iter=15)
                if f_nm < best_f:
                    best_x = x_nm
                    best_f = f_nm

        # Phase 4: If still budget left, try Powell from second-best candidate
        if calls < budget - 20 and time.time() < deadline - 10:
            # Generate a few more candidates
            candidates = []
            for i in range(n_global + 1, n_global + 30):
                if calls >= budget - 20 or time.time() > deadline - 10:
                    break
                x = halton(i, n)
                f = try_eval(x)
                if f is not None:
                    candidates.append((f, x))

            if candidates:
                candidates.sort(key=lambda t: t[0])
                for f_c, x_c in candidates:
                    if np.linalg.norm(x_c - best_x) > 1.5:
                        remaining = budget - calls
                        if remaining >= 30:
                            x_try, f_try = powell_method(x_c, f_c, max_iter=8)
                            if f_try < best_f:
                                best_x = x_try
                                best_f = f_try
                        break

        return best_x, best_f
