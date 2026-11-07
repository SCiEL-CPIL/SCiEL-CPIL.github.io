"""Newton-step optimizer exploiting quadratic structure of BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Newton-step optimizer exploiting quadratic structure of BBOB f11 Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def newton_step(x_center, h=0.05):
            """Fit local quadratic model via finite differences and compute Newton step."""
            nonlocal call_count, best_f, best_x
            if call_count >= budget - 2 * dim - 2:
                return None

            n = dim
            # Evaluate at center
            f0 = try_eval(x_center)
            if f0 is None:
                return None

            # Evaluate at x_center +/- h*e_i for gradient and Hessian diagonal
            grad = np.zeros(n)
            hess = np.zeros((n, n))
            for i in range(n):
                x_plus = x_center.copy()
                x_plus[i] += h
                x_minus = x_center.copy()
                x_minus[i] -= h
                fp = try_eval(x_plus)
                fm = try_eval(x_minus)
                if fp is None or fm is None:
                    return None
                grad[i] = (fp - fm) / (2 * h)
                hess[i, i] = (fp - 2 * f0 + fm) / (h ** 2)

            # Estimate off-diagonal Hessian entries using central differences
            # H_ij = (f(x+he_i+he_j) - f(x+he_i-he_j) - f(x-he_i+he_j) + f(x-he_i-he_j)) / (4*h^2)
            for i in range(n):
                for j in range(i + 1, n):
                    if call_count >= budget:
                        break
                    x_pp = x_center.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pm = x_center.copy()
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_mp = x_center.copy()
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mm = x_center.copy()
                    x_mm[i] -= h
                    x_mm[j] -= h

                    fpp = try_eval(x_pp)
                    fpm = try_eval(x_pm)
                    fmp = try_eval(x_mp)
                    fmm = try_eval(x_mm)
                    if fpp is None or fpm is None or fmp is None or fmm is None:
                        break

                    hij = (fpp - fpm - fmp + fmm) / (4 * h ** 2)
                    hess[i, j] = hij
                    hess[j, i] = hij

            if call_count >= budget:
                return None

            # Regularize Hessian for stability
            hess_reg = hess + 1e-6 * np.eye(n)

            # Solve H * dx = -grad
            try:
                dx = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(hess_reg, -grad, rcond=None)[0]
                except:
                    return None

            # Clip step to reasonable magnitude
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm * 1.0

            x_new = x_center + dx
            x_new = np.clip(x_new, -5.0, 5.0)
            return x_new

        def local_newton_polish(x_start, max_iters=10):
            """Iteratively apply Newton steps to converge to local minimum."""
            nonlocal call_count, best_f, best_x
            x = x_start.copy()
            for it in range(max_iters):
                if call_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break

                # Try Newton step with adaptive step size
                found = False
                for h in [0.1, 0.05, 0.02, 0.01]:
                    x_new = newton_step(x, h=h)
                    if x_new is None:
                        continue
                    fx = try_eval(x_new)
                    if fx is not None and fx < best_f - 1e-12:
                        x = x_new
                        found = True
                        break
                    elif fx is not None:
                        # Line search: try smaller step
                        alpha = 0.5
                        for _ in range(3):
                            x_trial = x + alpha * (x_new - x)
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            ft = try_eval(x_trial)
                            if ft is not None and ft < best_f - 1e-12:
                                x = x_trial
                                found = True
                                break
                            alpha *= 0.5
                        if found:
                            break
                if not found:
                    break

        # Phase 1: Global search to find good starting point
        # Use a combination of random sampling and structured search
        # The Discus function has a narrow valley, so we need good global coverage

        # First: evaluate at origin and a few structured points
        structured_points = [np.zeros(dim)]
        # Axis-aligned points at various distances
        for i in range(dim):
            for r in [1.0, 2.0, 3.0, 4.0]:
                for sign in [-1, 1]:
                    p = np.zeros(dim)
                    p[i] = sign * r
                    structured_points.append(p)
        # Random points
        for _ in range(50):
            structured_points.append(rng.uniform(-4.5, 4.5, size=dim))

        for p in structured_points:
            if call_count >= budget:
                break
            try_eval(p)

        # Phase 2: Refine best point with Newton steps
        if best_x is not None and call_count < budget:
            local_newton_polish(best_x, max_iters=20)

        # Phase 3: Try multiple local optimizations from diverse starting points
        # Generate candidate starting points around the best and in promising regions
        if call_count < budget:
            candidates = []
            # Points near best
            for scale in [0.5, 1.0, 2.0]:
                candidates.append(best_x + scale * rng.standard_normal(dim))
            # Points along axes from best
            for i in range(dim):
                for sign in [-1, 1]:
                    p = best_x.copy()
                    p[i] += sign * 1.0
                    candidates.append(p)
            # Random points in box
            for _ in range(30):
                candidates.append(rng.uniform(-4.5, 4.5, size=dim))

            for cand in candidates:
                if call_count >= budget:
                    break
                if time.time() > deadline - 30:
                    break
                cand = np.clip(cand, -5.0, 5.0)
                fc = try_eval(cand)
                if fc is None:
                    break
                # If this is a good point, try to polish it
                if fc < best_f + 5.0:  # within 5 units of best
                    local_newton_polish(cand, max_iters=10)

        # Phase 4: Final intensive Newton polish on the global best
        if best_x is not None and call_count < budget:
            local_newton_polish(best_x, max_iters=30)

        # Phase 5: If budget remains, try a few more exploratory Newton refinements
        while call_count < budget - 10:
            if time.time() > deadline - 20:
                break
            # Try a random perturbation of best and polish
            x_try = best_x + 0.3 * rng.standard_normal(dim)
            x_try = np.clip(x_try, -5.0, 5.0)
            ft = try_eval(x_try)
            if ft is None:
                break
            if ft < best_f + 2.0:
                local_newton_polish(x_try, max_iters=10)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
