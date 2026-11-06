"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150 s wall-clock kill
    MAX_EVALS = 500
    TOL = 1e-8

    eval_count = 0

    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    def remaining():
        return DEADLINE - (time.time() - start_time)

    # Track best point (smallest |f(x)|) found so far
    best_x = None
    best_fx = None

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if not math.isfinite(fx):
            return
        if best_x is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    x0 = float(x0)

    # First evaluation at x0
    try:
        fx0 = call_f(x0)
    except Exception:
        return x0
    update_best(x0, fx0)

    if abs(fx0) <= TOL:
        return x0

    if not math.isfinite(fx0):
        # f(x0) is inf/nan; try a small perturbation to get a finite value
        for dx in (1e-4, 1e-6, 1e-8, 1e-2, 1.0):
            xp = x0 + dx
            try:
                fxp = call_f(xp)
            except Exception:
                continue
            update_best(xp, fxp)
            if math.isfinite(fxp) and abs(fxp) <= TOL:
                return xp
            if math.isfinite(fxp):
                x0 = xp
                fx0 = fxp
                break
        if best_x is not None and abs(best_fx) <= TOL:
            return best_x
        return x0

    # Try to find a bracket [a, b] with a sign change.
    a, b = None, None
    fa, fb = None, None

    # First try a small symmetric expansion to find a sign change.
    scale = max(1.0, abs(x0))
    found_bracket = False
    for i in range(1, 60):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        step = scale * (2.0 ** i)
        for (lo, hi) in ((x0 - step, x0 + step),):
            try:
                flo = call_f(lo)
            except Exception:
                continue
            update_best(lo, flo)
            if not math.isfinite(flo):
                continue
            if abs(flo) <= TOL:
                return lo
            try:
                fhi = call_f(hi)
            except Exception:
                continue
            update_best(hi, fhi)
            if not math.isfinite(fhi):
                continue
            if abs(fhi) <= TOL:
                return hi
            if flo == 0.0:
                return lo
            if fhi == 0.0:
                return hi
            if flo * fhi < 0.0:
                a, b, fa, fb = lo, hi, flo, fhi
                found_bracket = True
                break
        if found_bracket:
            break

    # If no bracket found, fall back to a one-sided scan to find a sign change
    if not found_bracket:
        for i in range(1, 40):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            step = scale * (2.0 ** i)
            for (lo, hi) in ((x0, x0 + step), (x0 - step, x0)):
                try:
                    flo = call_f(lo)
                except Exception:
                    continue
                update_best(lo, flo)
                if not math.isfinite(flo):
                    continue
                if abs(flo) <= TOL:
                    return lo
                try:
                    fhi = call_f(hi)
                except Exception:
                    continue
                update_best(hi, fhi)
                if not math.isfinite(fhi):
                    continue
                if abs(fhi) <= TOL:
                    return hi
                if flo * fhi < 0.0:
                    a, b, fa, fb = lo, hi, flo, fhi
                    found_bracket = True
                    break
            if found_bracket:
                break

    # If we still have no bracket, use a local Newton/secant approach without
    # bracketing (best effort), but keep the best point.
    if not found_bracket:
        # Use secant method with two nearby points
        h = 1e-6 * max(1.0, abs(x0))
        try:
            x1 = x0 + h
            fx1 = call_f(x1)
        except Exception:
            return best_x if best_x is not None else x0
        update_best(x1, fx1)
        if abs(fx1) <= TOL:
            return x1
        if not math.isfinite(fx1):
            return best_x if best_x is not None else x0

        x_prev, f_prev = x0, fx0
        x_cur, f_cur = x1, fx1
        for _ in range(200):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(f_cur) <= TOL:
                return x_cur
            if x_cur == x_prev:
                break
            d = (f_cur - f_prev) / (x_cur - x_prev)
            if d == 0 or not math.isfinite(d):
                # Perturb to escape
                x_prev, f_prev = x_cur, f_cur
                x_cur = x_cur + 1e-4 * max(1.0, abs(x_cur))
                try:
                    f_cur = call_f(x_cur)
                except Exception:
                    break
                update_best(x_cur, f_cur)
                if abs(f_cur) <= TOL:
                    return x_cur
                continue
            x_new = x_cur - f_cur / d
            if not math.isfinite(x_new):
                break
            try:
                f_new = call_f(x_new)
            except Exception:
                break
            update_best(x_new, f_new)
            if abs(f_new) <= TOL:
                return x_new
            if not math.isfinite(f_new):
                break
            x_prev, f_prev = x_cur, f_cur
            x_cur, f_cur = x_new, f_new

        return best_x if best_x is not None else x0

    # We have a bracket [a, b] with fa*fb < 0. Use a hybrid
    # bisection + secant/Newton (Illinois-like / Brent-lite).
    # Maintain invariant: fa * fb < 0.
    for _ in range(300):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b

        # Secant (linear interpolation) estimate
        if fb != fa:
            x_sec = a - fa * (b - a) / (fb - fa)
        else:
            x_sec = 0.5 * (a + b)

        # Decide: if secant is inside the bracket and not too close to an
        # endpoint, use it; otherwise bisect.
        do_secant = (a < x_sec < b) and (abs(x_sec - a) > 1e-12 * max(1.0, abs(a))) and (abs(x_sec - b) > 1e-12 * max(1.0, abs(b)))
        if do_secant:
            x_mid = x_sec
        else:
            x_mid = 0.5 * (a + b)

        try:
            f_mid = call_f(x_mid)
        except Exception:
            break
        update_best(x_mid, f_mid)
        if abs(f_mid) <= TOL:
            return x_mid
        if not math.isfinite(f_mid):
            # Treat as bisection fallback: pick midpoint of the sub-interval
            # we would have chosen; just bisect.
            x_mid = 0.5 * (a + b)
            try:
                f_mid = call_f(x_mid)
            except Exception:
                break
            update_best(x_mid, f_mid)
            if abs(f_mid) <= TOL:
                return x_mid
            if not math.isfinite(f_mid):
                break

        if fa * f_mid < 0.0:
            b = x_mid
            fb = f_mid
        elif fb * f_mid < 0.0:
            a = x_mid
            fa = f_mid
        else:
            # f_mid == 0 handled above; otherwise, nudge: shrink toward the
            # side with the smaller |f| to keep progress.
            if abs(fa) < abs(fb):
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid

        # Stop if interval is tiny relative to tolerance
        if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            break

    # Return best point found
    if best_x is not None and abs(best_fx) <= TOL:
        return best_x
    # Prefer the endpoint of the bracket with smaller |f|
    if a is not None and b is not None:
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    return best_x if best_x is not None else x0
# EVOLVE-BLOCK-END
