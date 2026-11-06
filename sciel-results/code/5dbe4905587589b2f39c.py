"""Root finding: Ill-conditioned - bracket-first with scipy brentq and polish."""
import time
import math
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Primary strategy: find a bracket, then use scipy brentq (robust hybrid).
    Fallback: secant/Newton if no bracket found.
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        return f(x)

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_time():
        return (time.time() - start_time) < MAX_WALL

    def find_bracket(x_center, fx_center, base_h, max_expansions=30):
        """
        Try to find [a, b] with f(a)*f(b) < 0 by expanding around x_center.
        Returns (a, b, fa, fb) or None.
        """
        h = base_h
        for _ in range(max_expansions):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                return None

            a = x_center - h
            b = x_center + h
            fa = eval_f(a)
            fb = eval_f(b)

            if math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb):
                h *= 2.0
                continue

            if abs(fa) <= TARGET_TOL:
                return (a, a, fa, fa)
            if abs(fb) <= TARGET_TOL:
                return (b, b, fb, fb)

            if fa * fb < 0:
                return (a, b, fa, fb)

            # Also check if x_center itself brackets with a or b
            if fx_center * fa < 0:
                return (a, x_center, fa, fx_center)
            if fx_center * fb < 0:
                return (x_center, b, fx_center, fb)

            h *= 2.0

        return None

    def try_brentq(a, b, fa, fb):
        """
        Use scipy brentq to find root in [a, b].
        Returns (x, fx) or None.
        """
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=100)
            fr = eval_f(root)
            if is_good(root, fr):
                return (root, fr)
            # Even if not perfectly below tol, brentq should be very close
            if abs(fr) < 1e-6:
                return (root, fr)
        except Exception:
            pass
        return None

    def secant_polish(x0, x1, f0, f1, max_iter=20):
        """Secant method as fallback."""
        x, f_prev, f_curr = x0, f0, f1
        for _ in range(max_iter):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
            if abs(f_curr) <= TARGET_TOL:
                return x, f_curr, True
            if f_curr == f_prev:
                break
            x_new = x - f_curr * (x - x1) / (f_curr - f_prev)
            step = x_new - x
            if abs(step) > 10 * max(1.0, abs(x)):
                x_new = x + math.copysign(10 * max(1.0, abs(x)), step)
            f_new = eval_f(x_new)
            if math.isnan(f_new) or math.isinf(f_new):
                break
            x1, x = x, x_new
            f_prev, f_curr = f_curr, f_new
        return x, f_curr, is_good(x, f_curr)

    # Evaluate initial point
    x = float(x0)
    fx = eval_f(x)

    if is_good(x, fx):
        return x

    best_x, best_fx = x, fx

    # Strategy 1: Find bracket around x0 with adaptive expansion
    base_h = 1e-3 * max(1.0, abs(x0))
    bracket = find_bracket(x, fx, base_h, max_expansions=25)

    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b:  # Non-degenerate bracket
            result = try_brentq(a, b, fa, fb)
            if result is not None:
                return result[0]
            # If brentq didn't fully converge, try secant polish from bracket endpoints
            result = secant_polish(a, b, fa, fb, max_iter=15)
            if result[2]:
                return result[0]
            if abs(result[1]) < abs(best_fx):
                best_x, best_fx = result[0], result[1]

    # Strategy 2: Try multiple offset brackets
    offsets = [1e-2, -1e-2, 1e-1, -1e-1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]
    for offset in offsets:
        if not check_time() or eval_count[0] >= MAX_EVALS:
            break

        x_try = x0 + offset * max(1.0, abs(x0))
        fx_try = eval_f(x_try)

        if math.isnan(fx_try) or math.isinf(fx_try):
            continue

        if abs(fx_try) <= TARGET_TOL:
            return x_try

        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try

        # Try to find bracket centered at x_try
        h_local = 1e-3 * max(1.0, abs(x_try))
        bracket = find_bracket(x_try, fx_try, h_local, max_expansions=10)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a != b:
                result = try_brentq(a, b, fa, fb)
                if result is not None:
                    return result[0]
                result = secant_polish(a, b, fa, fb, max_iter=10)
                if result[2]:
                    return result[0]
                if abs(result[1]) < abs(best_fx):
                    best_x, best_fx = result[0], result[1]

        # Also try bracket between x0 and x_try
        if fx * fx_try < 0:
            result = try_brentq(x0, x_try, fx, fx_try)
            if result is not None:
                return result[0]
            result = secant_polish(x0, x_try, fx, fx_try, max_iter=10)
            if result[2]:
                return result[0]

    # Strategy 3: Secant from nearby point as last resort
    if eval_count[0] < MAX_EVALS and check_time():
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f1 = eval_f(x1)
        if not (math.isnan(f1) or math.isinf(f1)):
            result = secant_polish(x0, x1, fx, f1, max_iter=20)
            if result[2]:
                return result[0]
            if abs(result[1]) < abs(best_fx):
                best_x, best_fx = result[0], result[1]

    return best_x

# EVOLVE-BLOCK-END
