"""Root finder for 3 hidden smooth polynomial equations.
Design: robust Brent-style (bisection + secant/interpolation) with cheap
geometric bracketing, a strict call budget, a wall-clock deadline guard, and a
tight polish pass. Keeps the find_root(f, x0) -> float interface.
"""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start = time.monotonic()
    DEADLINE = 115.0          # wall-clock safety margin (evaluator kills at 150s)
    MAX_CALLS = 400           # strict budget (< 500 cap)
    TOL = 1e-12               # internal tolerance (well under required 1e-8)

    calls = 0
    best_x = None
    best_abs = math.inf

    def ok(t):
        return (time.monotonic() - start) < DEADLINE

    def call(x):
        nonlocal calls, best_x, best_abs
        calls += 1
        y = f(x)
        if isinstance(y, (int, float)):
            y = float(y)
        if math.isfinite(y):
            a = abs(y)
            if a < best_abs:
                best_abs = a
                best_x = float(x)
        return y

    x0 = float(x0)
    f0 = call(x0)
    if not math.isfinite(f0):
        return x0
    if abs(f0) <= TOL:
        return x0

    # --- Phase 1: cheap geometric bracketing around x0 -------------------
    scale = max(1.0, abs(x0))
    lo = hi = None
    flo = fhi = None
    a = x0 - scale
    b = x0 + scale
    fa = call(a)
    fb = call(b)
    if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
        lo, hi, flo, fhi = a, b, fa, fb
    else:
        # Expand outward geometrically until a sign change is found.
        d = scale
        for _ in range(30):
            if not ok() or calls >= MAX_CALLS - 60:
                break
            a = x0 - d
            b = x0 + d
            fa = call(a)
            fb = call(b)
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
                lo, hi, flo, fhi = a, b, fa, fb
                break
            d *= 2.0
            if d > 1e150:
                break

    if lo is not None:
        # --- Phase 2: Brent-style refinement on [lo, hi] ----------------
        # Invariant: flo*fhi < 0, x is the current best estimate.
        x = hi if abs(fhi) < abs(flo) else lo
        fx = fhi if x is hi else flo
        for _ in range(200):
            if not ok() or calls >= MAX_CALLS:
                break
            if abs(fx) <= TOL:
                break
            # Try secant/interpolation step; fall back to bisection.
            step = None
            if flo != fhi:
                # Inverse quadratic / secant estimate
                cand = x - fx * (hi - lo) / (fhi - flo)
                if lo < cand < hi and math.isfinite(cand):
                    step = cand
            if step is None:
                step = 0.5 * (lo + hi)
                if step == x:
                    break
            fs = call(step)
            if not math.isfinite(fs):
                # discard; bisect
                step = 0.5 * (lo + hi)
                fs = call(step)
                if not math.isfinite(fs):
                    break
            if abs(fs) <= TOL:
                x, fx = step, fs
                break
            # Update bracket
            if flo * fs < 0:
                hi, fhi = step, fs
            else:
                lo, flo = step, fs
            x, fx = step, fs
        # --- Phase 3: tight polish around the bracket midpoint ------------
        if lo is not None:
            # Use the two bracket endpoints as secant seeds for a fast polish.
            xa, ya = lo, flo
            xb, yb = hi, fhi
            for _ in range(40):
                if not ok() or calls >= MAX_CALLS:
                    break
                if abs(yb) <= TOL:
                    break
                if ya == yb:
                    break
                xc = xb - yb * (xb - xa) / (yb - ya)
                if not (lo < xc < hi) or not math.isfinite(xc):
                    xc = 0.5 * (lo + hi)
                    if xc == xb:
                        break
                yc = call(xc)
                if not math.isfinite(yc):
                    break
                if abs(yc) <= TOL:
                    break
                if ya * yc < 0:
                    xb, yb = xc, yc
                else:
                    xa, ya = xc, yc
            # Return the best tracked candidate (smallest |f|).
            if best_x is not None and math.isfinite(best_x):
                return best_x
            return x

    # --- Fallback: no bracket found. Try local Newton/secant polish. -----
    x = x0
    fx = f0
    # Secant with a small perturbation.
    h = 1e-5 * max(1.0, abs(x0))
    x1 = x0 + h
    f1 = call(x1)
    if math.isfinite(f1):
        for _ in range(80):
            if not ok() or calls >= MAX_CALLS:
                break
            if abs(fx) <= TOL:
                break
            if f1 == fx:
                break
            xn = x - fx * (x - x1) / (fx - f1)
            if not math.isfinite(xn):
                break
            fn = call(xn)
            if not math.isfinite(fn):
                break
            x1, f1 = x, fx
            x, fx = xn, fn
    if best_x is not None and math.isfinite(best_x):
        return best_x
    return x

# EVOLVE-BLOCK-END
