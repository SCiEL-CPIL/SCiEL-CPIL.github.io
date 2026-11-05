"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses:
    - Safeguarded Inverse Cubic Interpolation (SICI) with a bracket
    - Newton with central-difference derivative and damping
    - Secant method
    - Bisection for guaranteed convergence when a bracket is found
    - Adaptive random restarts
    - Deadline guard (return best-so-far before 150 s wall-clock limit)
    """
    deadline = time.time() + 138.0  # safety margin before 150 s kill
    max_evals = 480
    eval_count = [0]
    TOL = 1e-8

    def elapsed():
        return time.time() - (deadline - 138.0)

    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not (isinstance(val, (int, float, np.integer, np.floating))):
                return None
            val = float(val)
            if not math.isfinite(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, ArithmeticError):
            eval_count[0] += 1
            return None

    x0 = float(x0)
    best_x = x0
    best_fx = None

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is None:
            return
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    def done(x, fx):
        return fx is not None and abs(fx) <= TOL

    # ---------- Safeguarded Inverse Cubic Interpolation (SICI) ----------
    def sici(x0v, fx0, max_iter=60):
        """
        Maintain a bracket [a, b] with fa*fb < 0.
        Use inverse cubic interpolation on (f, x) to propose the next x.
        Fall back to bisection if interpolation leaves the bracket or is ill-defined.
        """
        # Build an initial bracket around x0v
        a = b = None
        fa = fb = None
        # First, try small perturbations
        base = max(1.0, abs(x0v))
        for h in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            if time.time() > deadline:
                return None
            a_try = x0v - h * base
            b_try = x0v + h * base
            fa_try = eval_f(a_try)
            fb_try = eval_f(b_try)
            if fa_try is not None and fb_try is not None and fa_try * fb_try < 0:
                a, fa = a_try, fa_try
                b, fb = b_try, fb_try
                break
            # If one endpoint is already a root, return it
            if fa_try is not None and abs(fa_try) <= TOL:
                return a_try
            if fb_try is not None and abs(fb_try) <= TOL:
                return b_try
        if a is None:
            # Expand outward from x0v
            step = 0.1 * base
            a = x0v
            fa = fx0
            for _ in range(25):
                if time.time() > deadline:
                    return None
                # Expand right
                b_try = a + step
                fb_try = eval_f(b_try)
                if fb_try is not None:
                    if fb_try * fa < 0:
                        b, fb = b_try, fb_try
                        break
                    # Keep the endpoint with smaller |f| as anchor
                    if abs(fb_try) < abs(fa):
                        a, fa = b_try, fb_try
                step *= 2
            if b is None:
                # Expand left
                step = 0.1 * base
                b = x0v
                fb = fx0
                for _ in range(25):
                    if time.time() > deadline:
                        return None
                    a_try = b - step
                    fa_try = eval_f(a_try)
                    if fa_try is not None:
                        if fa_try * fb < 0:
                            a, fa = a_try, fa_try
                            break
                        if abs(fa_try) < abs(fb):
                            b, fb = a_try, fa_try
                    step *= 2
            if a is None or b is None:
                return None
            if fa * fb > 0:
                return None

        # Now we have bracket [a, b]
        # For cubic interpolation, keep a history of up to 4 points
        # Start with the two endpoints
        pts = [(a, fa), (b, fb)]
        # Add midpoint for a third point
        mid = (a + b) / 2.0
        fm = eval_f(mid)
        if fm is None:
            return None
        if abs(fm) <= TOL:
            return mid
        if fm * fa < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        pts = [(a, fa), (b, fb)]

        # SICI loop
        for i in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b

            # Inverse cubic interpolation on (f, x)
            # We need 4 points; if we only have 2, use quadratic (3 points) or secant
            # For robustness, use the two endpoints and the last two interior points
            # Simplify: use secant if we only have 2 points, cubic if 4
            n_pts = len(pts)
            x_new = None
            if n_pts >= 4:
                # Use the last 4 points for inverse cubic interpolation
                p = pts[-4:]
                f1, x1 = p[0]
                f2, x2 = p[1]
                f3, x3 = p[2]
                f4, x4 = p[3]
                # Inverse cubic: find x such that f(x) = 0
                # Use Lagrange interpolation on (f_i, x_i)
                # x(f) = sum x_i * prod_{j != i} (f - f_j) / (f_i - f_j), evaluated at f=0
                # This is a polynomial in f; set f=0
                try:
                    x_new = 0.0
                    for i_idx in range(4):
                        fi = [f1, f2, f3, f4][i_idx]
                        xi = [x1, x2, x3, x4][i_idx]
                        term = xi
                        for j_idx in range(4):
                            if i_idx == j_idx:
                                continue
                            fj = [f1, f2, f3, f4][j_idx]
                            denom = fi - fj
                            if abs(denom) < 1e-300:
                                term = 0.0
                                break
                            # At f=0: (0 - fj) / (fi - fj) = -fj / (fi - fj)
                            term *= (-fj) / denom
                        if term != 0.0:
                            x_new += term
                except (ZeroDivisionError, OverflowError, ArithmeticError):
                    x_new = None

            if x_new is None or not math.isfinite(x_new):
                # Fall back to secant using the two endpoints
                if fa != fb:
                    x_new = b - fb * (b - a) / (fb - fa)
                else:
                    x_new = (a + b) / 2.0

            # Safeguard: if x_new is outside [a, b] or too close to an endpoint, use bisection
            eps = 1e-12 * max(1.0, abs(a), abs(b))
            if not (a + eps < x_new < b - eps):
                x_new = (a + b) / 2.0

            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) <= TOL:
                return x_new

            # Update bracket
            if fx_new * fa < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new

            # Update point history: keep the last 4 points
            pts.append((x_new, fx_new))
            if len(pts) > 4:
                pts.pop(0)

            # Convergence check
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break

        # Return the best point in the bracket
        if abs(fa) <= abs(fb):
            return a
        else:
            return b

    # ---------- Newton with central difference and damping ----------
    def newton(x0v, fx0, max_iter=25):
        x = x0v
        fx = fx0
        for i in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fx) <= TOL:
                return x

            # Central difference derivative
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            fx_minus = eval_f(x - h)
            if fx_minus is None:
                # Forward difference
                d = (fx_plus - fx) / h
            else:
                d = (fx_plus - fx_minus) / (2 * h)

            if not math.isfinite(d) or abs(d) < 1e-300:
                break

            step = fx / d
            # Damp the step
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
                # Try half step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break

        return x if abs(fx) <= TOL else None

    # ---------- Secant method ----------
    def secant(x1, fx1, x2, fx2, max_iter=20):
        for i in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fx1) <= TOL:
                return x1
            if abs(fx2) <= TOL:
                return x2
            if fx1 == fx2:
                break
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) <= TOL:
                return x_new
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        best = x1 if abs(fx1) <= abs(fx2) else x2
        best_f = fx1 if abs(fx1) <= abs(fx2) else fx2
        return best if abs(best_f) <= TOL else None

    # ---------- Bisection ----------
    def bisection(a, b, fa, fb, max_iter=60):
        for i in range(max_iter):
            if time.time() > deadline:
                break
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= TOL:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        return mid

    # ---------- Main driver ----------
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
    update_best(x0, fx0)
    if done(x0, fx0):
        return x0

    # Try SICI first
    result = sici(x0, fx0, max_iter=50)
    if result is not None:
        fx_res = eval_f(result)
        update_best(result, fx_res)
        if done(result, fx_res):
            return result

    # Try Newton
    result = newton(x0, fx0, max_iter=20)
    if result is not None:
        fx_res = eval_f(result)
        update_best(result, fx_res)
        if done(result, fx_res):
            return result

    # Try secant
    h = 1e-5 * max(1.0, abs(x0))
    x2 = x0 + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        update_best(x2, fx2)
        if done(x2, fx2):
            return x2
        result = secant(x0, fx0, x2, fx2, max_iter=15)
        if result is not None:
            fx_res = eval_f(result)
            update_best(result, fx_res)
            if done(result, fx_res):
                return result

    # Random restarts with increasing radius
    rng = np.random.RandomState(12345)
    base = max(1.0, abs(x0))
    for attempt in range(15):
        if time.time() > deadline:
            break
        radius = (10.0 ** attempt) * 1e-4 * base
        # 3 random points per radius
        for _ in range(3):
            if time.time() > deadline:
                break
            x_try = x0 + rng.normal(0, radius)
            fx_try = eval_f(x_try)
            if fx_try is None:
                continue
            update_best(x_try, fx_try)
            if done(x_try, fx_try):
                return x_try

            # Try SICI from this point
            result = sici(x_try, fx_try, max_iter=20)
            if result is not None:
                fx_res = eval_f(result)
                update_best(result, fx_res)
                if done(result, fx_res):
                    return result

            # Try Newton
            result = newton(x_try, fx_try, max_iter=10)
            if result is not None:
                fx_res = eval_f(result)
                update_best(result, fx_res)
                if done(result, fx_res):
                    return result

            # Try secant
            h2 = 1e-6 * max(1.0, abs(x_try))
            x_try2 = x_try + h2
            fx_try2 = eval_f(x_try2)
            if fx_try2 is not None:
                update_best(x_try2, fx_try2)
                if done(x_try2, fx_try2):
                    return x_try2
                result = secant(x_try, fx_try, x_try2, fx_try2, max_iter=10)
                if result is not None:
                    fx_res = eval_f(result)
                    update_best(result, fx_res)
                    if done(result, fx_res):
                        return result

    # Final: if we have a bracket, do bisection to squeeze
    # Try to find a bracket from the best points
    if best_fx is not None:
        # We don't have a guaranteed bracket, so just return best
        pass

    return best_x
# EVOLVE-BLOCK-END
