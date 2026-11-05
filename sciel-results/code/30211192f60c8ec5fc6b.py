import math
import time


def find_root(f, x0):
    TOL = 1e-8
    TOL_POLISH = 1e-10  # tighter tolerance for polish pass
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 138.0  # safety margin before the 150s kill

    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = math.inf

    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() >= DEADLINE:
            return None
        evals += 1
        try:
            val = float(f(x))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                AttributeError, ArithmeticError):
            return None
        if not math.isfinite(val):
            return None
        av = abs(val)
        if av < best_abs:
            best_abs = av
            best_x = float(x)
            best_f = val
        return val

    def is_good(x, fx):
        return fx is not None and abs(fx) <= TOL

    def is_close(x, fx):
        """Check if we're close enough to start a polish pass."""
        return fx is not None and abs(fx) <= 1e-5

    def safe_get_deriv(fx):
        try:
            d = f.deriv
            if callable(d):
                return d
        except Exception:
            pass
        try:
            d = f.derivative
            if callable(d):
                return d
        except Exception:
            pass
        return None

    def muller(x0, x1, x2):
        """Muller's method - robust, doesn't require sign changes."""
        x = float(x0)
        x1 = float(x1)
        x2 = float(x2)
        fx = call_f(x)
        fx1 = call_f(x1)
        fx2 = call_f(x2)
        if fx is None and fx1 is None and fx2 is None:
            return x, None
        # Use the two points with finite values
        if fx is None:
            x, fx = x1, fx1
            x1, fx1 = x2, fx2
        if fx1 is None:
            x1, fx1 = x2, fx2
            x2, fx2 = x, fx
        if fx2 is None:
            return x, fx
        if is_good(x, fx):
            return x, fx
        if is_good(x1, fx1):
            return x1, fx1
        if is_good(x2, fx2):
            return x2, fx2

        for _ in range(100):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            # Compute quadratic interpolation
            d1 = x1 - x
            d2 = x2 - x
            if abs(d1) < 1e-15 or abs(d2) < 1e-15:
                break
            # Lagrange coefficients
            a1 = (fx1 - fx) / d1
            a2 = (fx2 - fx) / d2
            a3 = (a2 - a1) / (d2 - d1)
            if not math.isfinite(a3) or abs(a3) < 1e-300:
                break
            # Solve for next step: a3*h^2 + a1*h + fx = 0
            disc = a1*a1 - 4*a3*fx
            if disc < 0:
                break
            sqrt_disc = math.sqrt(disc)
            h1 = (-a1 + sqrt_disc) / (2*a3)
            h2 = (-a1 - sqrt_disc) / (2*a3)
            # Choose the step with smaller magnitude
            h = h1 if abs(h1) < abs(h2) else h2
            if not math.isfinite(h):
                break
            # Damp large steps
            if abs(h) > 10.0 * max(1.0, abs(x)):
                h = math.copysign(10.0 * max(1.0, abs(x)), h)
            xn = x + h
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            if is_good(xn, fxn):
                return xn, fxn
            # Update points
            x, fx = xn, fxn
            x1, fx1 = x2, fx2
            x2, fx2 = xn, fxn
        if fx is not None and (fx1 is None or abs(fx) < abs(fx1)):
            return x, fx
        if fx1 is not None:
            return x1, fx1
        return x, fx

    def newton(x0):
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return x, None
        if is_good(x, fx):
            return x, fx
        
        best_x_local = x
        best_f_local = fx
        best_abs_local = abs(fx)

        deriv = safe_get_deriv(f)

        for _ in range(40):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            
            if deriv is not None:
                try:
                    d = float(deriv(x))
                    if not math.isfinite(d) or abs(d) < 1e-300:
                        break
                except Exception:
                    break
            else:
                h = 1e-6 * max(1.0, abs(x))
                fx2 = call_f(x + h)
                if fx2 is None:
                    break
                d = (fx2 - fx) / h
                if not math.isfinite(d) or abs(d) < 1e-15:
                    break

            step = fx / d
            if not math.isfinite(step):
                break
            xn = x - step
            # damp large steps
            if abs(xn - x) > 10.0 * max(1.0, abs(x)):
                xn = x + math.copysign(10.0 * max(1.0, abs(x)), xn - x)
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            
            if abs(fxn) < best_abs_local:
                best_x_local = xn
                best_f_local = fxn
                best_abs_local = abs(fxn)

            if is_good(xn, fxn):
                return xn, fxn
            
            if abs(fxn) < abs(fx):
                x = xn
                fx = fxn
            else:
                # try a half step
                xh = x - 0.5 * step
                fxh = call_f(xh)
                if fxh is not None and abs(fxh) < abs(fx):
                    x = xh
                    fx = fxh
                    if abs(fx) < best_abs_local:
                        best_x_local = x
                        best_f_local = fx
                        best_abs_local = abs(fx)
                else:
                    break
        return best_x_local, best_f_local

    def illinois(x0, x1):
        a = float(x0)
        b = float(x1)
        fa = call_f(a)
        fb = call_f(b)
        if fa is None or fb is None:
            if fa is not None:
                return a, fa
            if fb is not None:
                return b, fb
            return a, None
        if is_good(a, fa):
            return a, fa
        if is_good(b, fb):
            return b, fb

        for _ in range(200):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            if fa * fb > 0:
                break
            if is_good(b, fb):
                return b, fb
            # Illinois weighting: shrink the side with the larger magnitude
            if abs(fa) < abs(fb):
                # c is closer to a (root nearer a); replace b
                x = a + (b - a) * fa / (fa - fb)
                fx = call_f(x)
                if fx is None:
                    break
                if is_good(x, fx):
                    return x, fx
                b, fb = x, fx
            else:
                x = a + (b - a) * fa / (fa - fb)
                fx = call_f(x)
                if fx is None:
                    break
                if is_good(x, fx):
                    return x, fx
                a, fa = x, fx
            if abs(b - a) <= 2 * 2.22e-16 * max(1.0, abs(b)):
                break
        # return the better of the two endpoints
        if fa is not None and (fb is None or abs(fa) < abs(fb)):
            return a, fa
        if fb is not None:
            return b, fb
        return a, fa

    def secant(x0, x1):
        x = float(x0)
        xp = float(x1)
        fx = call_f(x)
        fxp = call_f(xp)
        if fx is None and fxp is None:
            return x, None
        if fx is None:
            x, fx = xp, fxp
        if fxp is None:
            xp, fxp = x, fx
        if is_good(x, fx):
            return x, fx
        if is_good(xp, fxp):
            return xp, fxp

        best_x_local = x if (fxp is None or abs(fx) < abs(fxp)) else xp
        best_f_local = fx if (fxp is None or abs(fx) < abs(fxp)) else fxp
        best_abs_local = abs(best_f_local)

        for _ in range(200):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            if fx == fxp:
                break
            xn = x - fx * (x - xp) / (fx - fxp)
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            if abs(fxn) < best_abs_local:
                best_x_local = xn
                best_f_local = fxn
                best_abs_local = abs(fxn)
            if is_good(xn, fxn):
                return xn, fxn
            # keep the two most recent distinct points
            xp, fxp = x, fx
            x, fx = xn, fxn
        return best_x_local, best_f_local

    x0 = float(x0)

    # 1) Newton (cheap, best near a good start)
    x, fx = newton(x0)
    if is_good(x, fx):
        return x
    # If close, do a polish pass
    if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        x, fx = newton(x)
        if is_good(x, fx):
            return x

    # 2) Secant from two nearby points
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 2:
        h = 1e-4 * max(1.0, abs(x0))
        x, fx = secant(x0, x0 + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # 3) Muller's method (robust, no sign change needed)
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        h = 1e-4 * max(1.0, abs(x0))
        x, fx = muller(x0, x0 + h, x0 - h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # 4) Illinois on progressively wider brackets (start small for ill-conditioned)
    for h in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3):
        if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 2:
            break
        x, fx = illinois(x0 - h, x0 + h)
        if is_good(x, fx):
            return x
        if fx is not None and abs(fx) < TOL:
            return x
        # If close, polish
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # 5) Secant from two far points
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 2:
        h = 1.0
        x, fx = secant(x0 - h, x0 + h)
        if is_good(x, fx):
            return x
        if is_close(x, fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
            x2, fx2 = newton(x)
            if is_good(x2, fx2):
                return x2

    # Final polish: if best is close, try one more Newton
    if is_close(best_x, best_f) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        x, fx = newton(best_x)
        if is_good(x, fx):
            return x

    # Return the globally best point found
    return best_x
