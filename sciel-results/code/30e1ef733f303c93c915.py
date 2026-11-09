"""
Optimizer for BBOB f1 (Sphere) with multi-scale coordinate descent.

Strategy:
1. Deterministic multi-scale coordinate descent sweeps: at each scale, perform
   1D line searches (ternary) along each coordinate with step sizes
   exponentially decreasing. This exploits the separable quadratic structure.
2. A Newton polish using a centered quadratic model fit at the end.
3. Robust deadline guard and budget tracking.
"""

import numpy as np
import time


class Optimizer:
    """
    Deterministic multi-scale coordinate descent + Newton polish for Sphere.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330 s kill

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def now_ok():
            return time.time() < deadline

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        call_count = 0

        def safe_eval(x):
            nonlocal call_count
            x = np.asarray(x, dtype=np.float64)
            call_count += 1
            return float(func(x))

        # ------------------------------------------------------------------
        # 1-D ternary search along coordinate i at point x with step delta
        # minimizes f over x[i] in {x[i] - delta, ..., x[i] + delta}
        # Returns (new_x, new_f, improved)
        # Cost: up to 6 evaluations (4 per ternary iteration, 2 iters)
        # ------------------------------------------------------------------
        def coordinate_search(x, fx, i, delta, max_calls=6):
            nonlocal call_count
            if call_count + max_calls > budget:
                return x, fx, False

            def f_1d(val):
                nonlocal call_count
                x_t = x.copy()
                x_t[i] = val
                call_count += 1
                return safe_eval(x_t)  # careful: safe_eval already increments
            # Actually, let's inline to avoid double counting.
            # Redefine f_1d without safe_eval.
            def f_1d(val):
                nonlocal call_count
                x_t = x.copy()
                x_t[i] = val
                call_count += 1
                return func(x_t)

            lo = x[i] - delta
            hi = x[i] + delta
            if lo < -5.0:
                lo = -5.0
            if hi > 5.0:
                hi = 5.0
            if hi - lo < 1e-14:
                return x, fx, False

            # Ternary search, 2 iterations
            for _ in range(2):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                f1 = f_1d(m1)
                f2 = f_1d(m2)
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1

            best_val = (lo + hi) / 2.0
            f_best = f_1d(best_val)

            x_new = x.copy()
            x_new[i] = best_val
            improved = f_best < fx
            if improved:
                return x_new, f_best, True
            else:
                return x, fx, False

        # ------------------------------------------------------------------
        # Multi-scale coordinate descent
        # ------------------------------------------------------------------
        x = np.zeros(dim)
        fx = safe_eval(x)
        best_x = x.copy()
        best_fx = fx

        # Initial coarse scales: from 5.0 (full box half-width) down to 0.1
        # Each scale: sweep all coordinates
        # Budget per scale: up to 6*dim calls (2 iterations * 3 evals per coord)
        # 8 scales * 5 coords * 6 = 240 calls, leaves room for polish
        # We'll use scales: 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625
        # That's 8 scales. 8*5*6 = 240 calls. Plus initial = 241.
        # Leaves ~259 for polish.

        scales = [5.0 * (0.5 ** k) for k in range(8)]  # 5.0, 2.5, ..., ~0.039

        for delta in scales:
            if not now_ok() or call_count >= budget:
                break
            for i in range(dim):
                if not now_ok() or call_count >= budget:
                    break
                x, fx, improved = coordinate_search(x, fx, i, delta, max_calls=6)
                if improved:
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

        # ------------------------------------------------------------------
        # Newton polish with centered quadratic model
        # ------------------------------------------------------------------
        # Use remaining budget for a few Newton steps.
        # Cost per Newton step: 2*dim + 1 = 11 calls
        # We can afford ~20 steps if we have 220 calls left.

        x_pol = best_x.copy()
        fx_pol = best_fx

        # We'll try up to 15 Newton steps
        for step in range(15):
            if not now_ok() or call_count >= budget - 2 * dim - 1:
                break

            # Estimate gradient via central differences
            eps = 1e-6
            grad = np.zeros(dim)
            for i in range(dim):
                x_p = x_pol.copy()
                x_m = x_pol.copy()
                x_p[i] += eps
                x_m[i] -= eps
                fp = func(x_p)
                fm = func(x_m)
                call_count += 2
                grad[i] = (fp - fm) / (2 * eps)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break

            # Estimate Hessian via central differences of gradient
            # For Sphere, Hessian is constant diagonal (2 * diag) if unrotated,
            # but with rotation it's full. Use central diff of grad.
            # Cost: 2*dim * dim = 50 calls per Hessian. That's expensive.
            # Instead, use a 1D curvature estimate per coordinate (diagonal Hessian).
            # For Sphere with rotation, diagonal Hessian is a good approximation.
            hess_diag = np.zeros(dim)
            for i in range(dim):
                x_pp = x_pol.copy()
                x_pm = x_pol.copy()
                x_mp = x_pol.copy()
                x_mm = x_pol.copy()
                h = 1e-5
                x_pp[i] += h
                x_pm[i] -= h
                x_mp[i] += h
                x_mm[i] -= h
                # f(x+he_i) - 2f(x) + f(x-he_i) / h^2
                f_pp = func(x_pp)
                f_pm = func(x_pm)
                f_mp = func(x_mp)
                f_mm = func(x_mm)
                call_count += 4
                # Actually, central second difference:
                # (f(x+h e_i) - 2 f(x) + f(x-h e_i)) / h^2
                # But we need f(x) which is fx_pol.
                hess_diag[i] = (f_pp - 2 * fx_pol + f_pm) / (h * h)
                # Wait, x_pp has +h, x_pm has -h. So f_pp = f(x + h e_i), f_pm = f(x - h e_i)
                # That's correct.

            # Newton step: x_new = x - H^{-1} grad, using diagonal Hessian
            # For positive definite Hessian (Sphere is convex), this is safe.
            # Clip step to avoid overshooting.
            alpha = 1.0
            for _ in range(5):
                step_vec = grad / hess_diag  # H^{-1} grad
                # Clip to avoid huge steps
                step_norm = np.linalg.norm(step_vec)
                if step_norm > 1.0:
                    step_vec = step_vec / step_norm  # limit step to unit length
                x_new = x_pol - alpha * step_vec
                x_new = clip_box(x_new)
                f_new = func(x_new)
                call_count += 1
                if f_new < fx_pol:
                    x_pol = x_new
                    fx_pol = f_new
                    if fx_pol < best_fx:
                        best_fx = fx_pol
                        best_x = x_pol.copy()
                    break
                else:
                    alpha *= 0.5
                    if alpha < 1e-8:
                        break

            if alpha < 1e-8:
                break

        return best_x, best_fx


# EVOLVE-BLOCK-END
