"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 110.0  # 110s safety margin before the 150s kill

    # Explicit call budget: stay comfortably under the 500-call per-equation limit.
    MAX_CALLS = 480
    call_count = 0

    def time_up():
        return time.time() > deadline

    def call_budget():
        return call_count >= MAX_CALLS

    def ok():
        return not time_up() and not call_budget()

    def feval(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    best_x = x
    best_abs = math.inf

    # Evaluate starting point
    best_fx = feval(x)
    if not math.isfinite(best_fx):
        # Can't trust the value; return x as best-effort.
        return x
    best_abs = abs(best_fx)
    if best_abs <= 1e-8:
        return x

    def consider(xv, fxv):
        nonlocal best_x, best_abs
        if math.isfinite(fxv):
            a = abs(fxv)
            if a < best_abs:
                best_x, best_abs = xv, a
                if a <= 1e-8:
                    return True
        return False

    # ---- Phase 1: Secant with bisection safeguard ----
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x
    x_curr = x + h
    fx_prev = best_fx
    fx_curr = feval(x_curr)
    if not math.isfinite(fx_curr):
        x_curr = x - h
        fx_curr = feval(x_curr)
        if not math.isfinite(fx_curr):
            return best_x
    if consider(x_curr, fx_curr):
        return x_curr

    for _ in range(30):
        if not ok():
            break
        if fx_prev == fx_curr or abs(fx_curr - fx_prev) < 1e-30:
            break
        denom = fx_curr - fx_prev
        x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
        if not math.isfinite(x_next):
            break
        # Bisection safeguard: if the secant step is huge, use midpoint instead.
        if abs(x_next - x_curr) > 10.0 * max(1.0, abs(x_curr)):
            x_next = 0.5 * (x_prev + x_curr)
        fx_next = feval(x_next)
        if not math.isfinite(fx_next):
            break
        if consider(x_next, fx_next):
            return x_next
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_next, fx_next

    if not ok():
        return best_x

    # ---- Phase 2: Find a bracket (cheap, symmetric expansion) ----
    a = best_x - 1.0
    b = best_x + 1.0
    fa = feval(a)
    fb = feval(b)
    found_bracket = False
    if math.isfinite(fa) and math.isfinite(fb):
        if consider(a, fa):
            return a
        if consider(b, fb):
            return b
        if fa * fb <= 0:
            found_bracket = True

    if not found_bracket and ok():
        step = 1.0
        for _ in range(40):
            if not ok():
                break
            a = best_x - step
            b = best_x + step
            fa = feval(a)
            fb = feval(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                step *= 2.0
                continue
            if consider(a, fa):
                return a
            if consider(b, fb):
                return b
            if fa * fb <= 0:
                found_bracket = True
                break
            step *= 2.0
            if step > 1e12:
                break

    # ---- Phase 3: Brent's method (fast, guaranteed) if bracketed ----
    if found_bracket and ok():
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=60)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # ---- Phase 4: Manual bisection fallback if bracketed ----
    if found_bracket and ok():
        for _ in range(60):
            if not ok():
                break
            mid = 0.5 * (a + b)
            fm = feval(mid)
            if not math.isfinite(fm):
                break
            if consider(mid, fm):
                return mid
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm

    # ---- Phase 5: Final high-precision polish from best_x ----
    if ok():
        x = best_x
        fx = feval(x)
        if not math.isfinite(fx):
            return best_x
        if consider(x, fx):
            return x
        for _ in range(15):
            if not ok():
                break
            h = 1e-6 * max(1.0, abs(x))
            x2 = x + h
            fx2 = feval(x2)
            if not math.isfinite(fx2):
                break
            if consider(x2, fx2):
                return x2
            if fx == fx2:
                break
            x_new = x - fx * (x - x2) / (fx - fx2)
            if not math.isfinite(x_new):
                break
            x, fx = x_new, feval(x_new)
            if consider(x, fx):
                return x

    return best_x
# EVOLVE-BLOCK-END
