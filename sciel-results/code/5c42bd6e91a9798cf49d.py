"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Core strategy: Illinois algorithm for guaranteed bracketed convergence,
    with Newton/secant for speed when derivatives are reliable, and
    multi-scale exploration for bracket discovery.
    """
    deadline = time.monotonic() + 140.0  # safety margin before 150s kill
    max_evals = 480
    eval_count = [0]
    TOL = 1e-8

    best_x = float(x0)
    best_fx = None

    def eval_f(x):
        nonlocal best_x, best_fx
        if time.monotonic() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            val = float(val)
            if not math.isfinite(val):
                return None
            if best_fx is None or abs(val) < abs(best_fx):
                best_fx = val
                best_x = x
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, ArithmeticError):
            eval_count[0] += 1
            return None

    def done(x, fx):
        return fx is not None and abs(fx) <= TOL

    # ---------- Illinois Algorithm (guaranteed convergence for bracketed root) ----------
    def illinois(a, fa, b, fb, max_iter=80):
        """
        Illinois variant of bisection: when one endpoint has smaller |f|,
        halve the OTHER endpoint's interval instead of the midpoint.
        This converges faster than plain bisection while maintaining the bracket.
        """
        for _ in range(max_iter):
            if time.monotonic() > deadline:
                break
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b

            # Illinois: shrink the side with larger |f|
            if abs(fa) < abs(fb):
                # Shrink the b-side: new point is midpoint of [a, b], but weight toward a
                # Actually Illinois: c = (a+b)/2, then if |f(a)| < |f(b)|, set b = c and f(b) = f(b)/2
                # Standard Illinois:
                mid = (a + b) / 2.0
                fm = eval_f(mid)
                if fm is None:
                    break
                if abs(fm) <= TOL:
                    return mid
                if fm * fa < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                # Halve the f-value of the endpoint with larger |f| to bias future midpoints
                if abs(fa) < abs(fb):
                    fb *= 0.5
                else:
                    fa *= 0.5
            else:
                mid = (a + b) / 2.0
                fm = eval_f(mid)
                if fm is None:
                    break
                if abs(fm) <= TOL:
                    return mid
                if fm * fa < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                if abs(fa) < abs(fb):
                    fb *= 0.5
                else:
                    fa *= 0.5

            # Convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break

        # Return best
        if abs(fa) <= abs(fb):
            return a
        else:
            return b

    # ---------- Find a bracket starting from x0 ----------
    def find_bracket(x0v, fx0):
        """
        Expand outward from x0v to find a sign change.
        Try multiple scales.
        """
        base = max(1.0, abs(x0v))
        scales = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

        for h in scales:
            if time.monotonic() > deadline:
                return None
            a_try = x0v - h * base
            b_try = x0v + h * base
            fa_try = eval_f(a_try)
            fb_try = eval_f(b_try)
            if fa_try is not None and fb_try is not None:
                if fa_try * fb_try < 0:
                    return (a_try, fa_try, b_try, fb_try)
                if abs(fa_try) <= TOL:
                    return (a_try, fa_try, a_try, fa_try)
                if abs(fb_try) <= TOL:
                    return (b_try, fb_try, b_try, fb_try)

        # Asymmetric expansion
        step = 0.1 * base
        a = x0v
        fa = fx0
        for _ in range(20):
            if time.monotonic() > deadline:
                return None
            b_try = a + step
            fb_try = eval_f(b_try)
            if fb_try is not None:
                if fb_try * fa < 0:
                    return (a, fa, b_try, fb_try)
                if abs(fb_try) <= TOL:
                    return (b_try, fb_try, b_try, fb_try)
            step *= 2

        step = 0.1 * base
        b = x0v
        fb = fx0
        for _ in range(20):
            if time.monotonic() > deadline:
                return None
            a_try = b - step
            fa_try = eval_f(a_try)
            if fa_try is not None:
                if fa_try * fb < 0:
                    return (a_try, fa_try, b, fb)
                if abs(fa_try) <= TOL:
                    return (a_try, fa_try, a_try, fa_try)
            step *= 2

        return None

    # ---------- Newton with damping ----------
    def newton(x0v, fx0, max_iter=20):
        x = x0v
        fx = fx0
        for _ in range(max_iter):
            if time.monotonic() > deadline:
                break
            if abs(fx) <= TOL:
                return x

            h = 1e-6 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            fx_minus = eval_f(x - h)
            if fx_minus is not None:
                d = (fx_plus - fx_minus) / (2 * h)
            else:
                d = (fx_plus - fx) / h

            if not math.isfinite(d) or abs(d) < 1e-300:
                break

            step = fx / d
            max_step = 0.5 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)

            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is not None and abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break

        return x if abs(fx) <= TOL else None

    # ---------- Secant ----------
    def secant(x1, fx1, x2, fx2, max_iter=20):
        for _ in range(max_iter):
            if time.monotonic() > deadline:
                break
            if abs(fx1) <= TOL:
                return x1
            if abs(fx2) <= TOL:
                return x2
            if fx1 == fx2:
                break
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            if not math.isfinite(x_new):
                break
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) <= TOL:
                return x_new
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        best = x1 if abs(fx1) <= abs(fx2) else x2
        return best if abs(fx1 if best == x1 else fx2) <= TOL else None

    # ---------- Main driver ----------
    x0 = float(x0)
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
    if done(x0, fx0):
        return x0

    # Phase 1: Try to find a bracket and use Illinois
    bracket = find_bracket(x0, fx0)
    if bracket is not None:
        a, fa, b, fb = bracket
        if a != b and fa * fb < 0:
            result = illinois(a, fa, b, fb, max_iter=70)
            if result is not None:
                fx_res = eval_f(result)
                if done(result, fx_res):
                    return result
        elif abs(fa) <= TOL:
            return a
        elif abs(fb) <= TOL:
            return b

    # Phase 2: Newton from x0
    result = newton(x0, fx0, max_iter=15)
    if result is not None:
        fx_res = eval_f(result)
        if done(result, fx_res):
            return result

    # Phase 3: Secant from x0
    h = 1e-5 * max(1.0, abs(x0))
    x2 = x0 + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        if done(x2, fx2):
            return x2
        result = secant(x0, fx0, x2, fx2, max_iter=15)
        if result is not None:
            fx_res = eval_f(result)
            if done(result, fx_res):
                return result

    # Phase 4: Multi-scale random exploration with Illinois
    rng = np.random.RandomState(42)
    base = max(1.0, abs(x0))
    radii = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for radius in radii:
        if time.monotonic() > deadline or eval_count[0] >= max_evals - 10:
            break
        for _ in range(3):
            if time.monotonic() > deadline or eval_count[0] >= max_evals - 10:
                break
            x_try = x0 + rng.normal(0, radius * base)
            fx_try = eval_f(x_try)
            if fx_try is None:
                continue
            if done(x_try, fx_try):
                return x_try

            # Try to bracket from this point
            bracket = find_bracket(x_try, fx_try)
            if bracket is not None:
                a, fa, b, fb = bracket
                if a != b and fa * fb < 0:
                    result = illinois(a, fa, b, fb, max_iter=30)
                    if result is not None:
                        fx_res = eval_f(result)
                        if done(result, fx_res):
                            return result
                elif abs(fa) <= TOL:
                    return a
                elif abs(fb) <= TOL:
                    return b

            # Quick Newton
            result = newton(x_try, fx_try, max_iter=8)
            if result is not None:
                fx_res = eval_f(result)
                if done(result, fx_res):
                    return result

    # Phase 5: Final sweep - try to bracket around best_x
    if best_fx is not None:
        bracket = find_bracket(best_x, best_fx)
        if bracket is not None:
            a, fa, b, fb = bracket
            if a != b and fa * fb < 0:
                result = illinois(a, fa, b, fb, max_iter=40)
                if result is not None:
                    fx_res = eval_f(result)
                    if done(result, fx_res):
                        return result

    return best_x
# EVOLVE-BLOCK-END
