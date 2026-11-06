"""Root finding: Ill-conditioned - Illinois method with adaptive bracketing and polish."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Illinois method (modified bisection) with adaptive bracket finding,
    plus secant polish for final precision.
    """
    start_time = time.time()
    MAX_WALL = 110.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_time():
        return time.time() - start_time < MAX_WALL

    def check_budget():
        return eval_count[0] < MAX_EVALS and check_time()

    # Illinois method: modified bisection that shrinks the larger sub-interval
    def illinois_method(a, b, fa, fb, max_iter=100):
        """
        Illinois method: like bisection but when the new point has the same sign
        as one endpoint, we shrink that endpoint's weight, causing the next
        midpoint to be closer to the other endpoint. This gives superlinear
        convergence for well-behaved functions while retaining bisection's
        guaranteed convergence.
        """
        if not check_budget():
            return None
        if fa * fb > 0:
            return None

        # Weights: when we keep an endpoint, we halve its weight
        wa = 1.0
        wb = 1.0
        left, right = a, b
        fl, fr = fa, fb
        best_x_local = left if abs(fl) < abs(fr) else right
        best_f_local = min(abs(fl), abs(fr))

        for _ in range(max_iter):
            if not check_budget():
                break
            # Weighted midpoint (Illinois)
            total_w = wa + wb
            if total_w <= 0:
                mid = (left + right) / 2.0
            else:
                mid = (wa * right + wb * left) / total_w

            fm = eval_f(mid)
            if is_good(mid, fm):
                return mid
            if abs(fm) < best_f_local:
                best_f_local = abs(fm)
                best_x_local = mid

            if fl * fm <= 0:
                # Root is in [left, mid]
                right = mid
                fr = fm
                # Shrink the weight of the endpoint we're moving toward
                wa *= 0.5
            else:
                # Root is in [mid, right]
                left = mid
                fl = fm
                wb *= 0.5

            if abs(right - left) < 1e-14 * max(1.0, abs(mid)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    # Secant polish: a few iterations to tighten the root
    def try_polish(x0, fx0, max_iter=15):
        """Secant method polish starting from a near-root."""
        if not check_budget():
            return None
        x = float(x0)
        fx = fx0
        if is_good(x, fx):
            return x

        # Get a second point close by
        h = 1e-5 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev

        for _ in range(max_iter):
            if not check_budget():
                break
            if fx == fx_prev:
                break
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            if abs(x_new - x_prev) < 1e-14 * max(1.0, abs(x_new)):
                break

        return x if abs(fx) <= TARGET_TOL else None

    # Newton with 5-point derivative and damping
    def try_newton(x0, max_iter=12):
        """Newton's method with 5-point FD derivative and damping."""
        if not check_budget():
            return None
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x

        best_x_local = x
        best_f_local = abs(fx)

        for _ in range(max_iter):
            if not check_budget():
                break

            # 2-point central difference
            h = (1e-4) ** 0.25 * max(1.0, abs(x))
            h = max(h, 1e-5)

            fx_m1 = eval_f(x - h)
            if is_good(x - h, fx_m1):
                return x - h
            fx_p1 = eval_f(x + h)
            if is_good(x + h, fx_p1):
                return x + h

            d = (fx_p1 - fx_m1) / (2*h)
            if d == 0 or not math.isfinite(d):
                break

            dx = fx / d
            if not math.isfinite(dx):
                break

            # Limit step
            max_step = 0.5 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)

            x_new = x - dx
            if not math.isfinite(x_new):
                break

            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new

            if abs(fx_new) < best_f_local:
                best_f_local = abs(fx_new)
                best_x_local = x_new

            # Damping
            if abs(fx_new) >= abs(fx):
                improved = False
                for damp in [0.5, 0.25, 0.125, 0.0625]:
                    if not check_budget():
                        break
                    x_d = x - damp * (x - x_new)
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        if abs(fx) < best_f_local:
                            best_f_local = abs(fx)
                            best_x_local = x
                        improved = True
                        break
                if not improved:
                    break
            else:
                x, fx = x_new, fx_new

            if abs(dx) < 1e-12 * max(1.0, abs(x)):
                break

        return best_x_local if best_f_local <= TARGET_TOL else None

    # Try to find a bracket [a, b] where f(a)*f(b) < 0
    def find_bracket(x0):
        """Adaptive bracket finding around x0."""
        if not check_budget():
            return None

        fx0 = eval_f(x0)
        if is_good(x0, fx0):
            return (x0, x0, fx0, fx0)

        # Try expanding in both directions
        base_h = 1e-2 * max(1.0, abs(x0))

        for scale in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
            if not check_budget():
                break
            h = base_h * scale
            xl = x0 - h
            xr = x0 + h

            fxl = eval_f(xl)
            if is_good(xl, fxl):
                return (xl, xl, fxl, fxl)

            fxr = eval_f(xr)
            if is_good(xr, fxr):
                return (xr, xr, fxr, fxr)

            if fxl * fxr < 0:
                return (xl, xr, fxl, fxr)

            if fxl * fx0 < 0:
                return (xl, x0, fxl, fx0)
            if fx0 * fxr < 0:
                return (x0, xr, fx0, fxr)

        # Try asymmetric brackets
        for sign in [1, -1]:
            for scale in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 500000]:
                if not check_budget():
                    break
                h = base_h * scale
                x2 = x0 + sign * h
                fx2 = eval_f(x2)
                if is_good(x2, fx2):
                    return (x2, x2, fx2, fx2)
                if fx0 * fx2 < 0:
                    return (x0, x2, fx0, fx2)

        # Try very wide brackets
        for scale in [1e6, 1e7, 1e8, 1e9, 1e10]:
            if not check_budget():
                break
            h = max(1.0, abs(x0)) * scale
            xl = x0 - h
            xr = x0 + h
            fxl = eval_f(xl)
            if is_good(xl, fxl):
                return (xl, xl, fxl, fxl)
            fxr = eval_f(xr)
            if is_good(xr, fxr):
                return (xr, xr, fxr, fxr)
            if fxl * fxr < 0:
                return (xl, xr, fxl, fxr)

        return None

    # Try scipy brentq as a fast path when bracketed
    def try_brentq(a, b, fa, fb):
        """Use scipy.optimize.brentq with our counted eval."""
        if not check_budget():
            return None
        try:
            from scipy.optimize import brentq
            def f_wrap(x):
                return eval_f(x)
            root = brentq(f_wrap, a, b, xtol=1e-13, rtol=1e-13, maxiter=100)
            fr = eval_f(root)
            if is_good(root, fr):
                return root
            if abs(fr) < 1e-6:
                return try_polish(root, fr)
            return None
        except Exception:
            return None

    # === MAIN STRATEGY ===

    # Step 1: Try to find a bracket and use Illinois or brentq
    bracket = find_bracket(x0)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b:
            return a
        if fa * fb < 0:
            # Try brentq first (faster), then Illinois as fallback
            result = try_brentq(a, b, fa, fb)
            if result is not None:
                return result
            # Illinois method as robust fallback
            result = illinois_method(a, b, fa, fb)
            if result is not None:
                return result

    # Step 2: Try Newton from x0
    result = try_newton(x0)
    if result is not None:
        return result

    # Step 3: Try secant from x0
    if check_budget():
        x = float(x0)
        fx = eval_f(x)
        if not is_good(x, fx):
            h = 1e-4 * max(1.0, abs(x))
            x_prev = x - h
            fx_prev = eval_f(x_prev)
            if not is_good(x_prev, fx_prev):
                for _ in range(30):
                    if not check_budget():
                        break
                    if fx == fx_prev:
                        break
                    d = (fx - fx_prev) / (x - x_prev)
                    if d == 0 or not math.isfinite(d):
                        break
                    x_new = x - fx / d
                    if not math.isfinite(x_new):
                        break
                    fx_new = eval_f(x_new)
                    if is_good(x_new, fx_new):
                        return x_new
                    x_prev, fx_prev = x, fx
                    x, fx = x_new, fx_new
                if abs(fx) <= TARGET_TOL:
                    return x

    # Step 4: Try Newton from perturbed points
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if not check_budget():
            break
        xp = x0 + offset * max(1.0, abs(x0))
        result = try_newton(xp, max_iter=10)
        if result is not None:
            return result

    # Step 5: Try to find a bracket from perturbed points
    for offset in [1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if not check_budget():
            break
        xp = x0 + offset * max(1.0, abs(x0))
        bracket = find_bracket(xp)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a == b:
                return a
            if fa * fb < 0:
                result = try_brentq(a, b, fa, fb)
                if result is not None:
                    return result
                result = illinois_method(a, b, fa, fb)
                if result is not None:
                    return result

    # Step 6: Local scan for sign change
    if check_budget():
        base_h = 1e-3 * max(1.0, abs(x0))
        prev_x = x0 - base_h
        prev_fx = eval_f(prev_x)
        for i in range(1, 80):
            if not check_budget():
                break
            cur_x = x0 + (i - 1) * base_h
            cur_fx = eval_f(cur_x)
            if is_good(cur_x, cur_fx):
                return cur_x
            if prev_fx * cur_fx < 0:
                result = try_brentq(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
                result = illinois_method(prev_x, cur_x, prev_fx, cur_fx)
                if result is not None:
                    return result
            prev_x, prev_fx = cur_x, cur_fx

    # Return best found
    return best_x[0]

# EVOLVE-BLOCK-END
