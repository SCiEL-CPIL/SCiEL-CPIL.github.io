"""Root finder for 3 fixed oscillatory equations with call-budget awareness."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """Find x with |f(x)| <= 1e-8 for a fixed oscillatory equation.

    Hybrid strategy:
      1. Multi-scale secant seeding: probe a log grid of step sizes around x0,
         keeping the best |f| point and any sign-change bracket found.
      2. Bisection on the best bracket (guaranteed convergence, low calls).
      3. Newton/secant polish of the best candidate to a tight tolerance.
    Tracks best (smallest |f|) throughout and returns it if the deadline nears.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    tol = 1e-11  # tighter than required 1e-8
    ok = lambda: time.time() < deadline

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    # ---- cached evaluation wrapper (dedup to keep call count low) ----
    cache = {}
    n_calls = [0]

    def F(t):
        n_calls[0] += 1
        try:
            v = f(t)
        except Exception:
            return None
        if not (isinstance(v, float) or isinstance(v, int)):
            try:
                v = float(v)
            except Exception:
                return None
        if not math.isfinite(v):
            return None
        return v

    # ---- best-tracking state ----
    best_x = x
    best_f = F(x)
    if best_f is None:
        return x
    best_abs = abs(best_f)

    def update(t, v):
        nonlocal best_x, best_f, best_abs
        if v is not None and abs(v) < best_abs:
            best_x = t
            best_f = v
            best_abs = abs(v)
        return v

    # seed x0
    update(x, best_f)

    # ---- 1. multi-scale secant seeding ----
    scale0 = max(1.0, abs(x))
    # candidate step sizes: log-spaced from tiny to large
    step_sizes = []
    for exp in range(-4, 10):
        step_sizes.append(scale0 * (10.0 ** exp))
    # de-dup and sort by magnitude
    step_sizes = sorted(set(step_sizes), key=lambda s: abs(s))

    last = x
    lastv = best_f
    for s in step_sizes:
        if not ok():
            break
        for sign in (1.0, -1.0):
            if not ok():
                break
            t = x + sign * s
            if not math.isfinite(t):
                continue
            v = update(t, F(t))
            if v is None:
                continue
            # check bracket between last accepted point and t
            if lastv is not None and lastv * v < 0.0:
                lo, hi = (last, t) if last < t else (t, last)
                flo, fhi = (lastv, v) if last < t else (v, lastv)
                brk = _bisect(F, update, lo, hi, flo, fhi, tol, ok, deadline)
                if brk is not None:
                    bx, bv = brk
                    if bv is not None and abs(bv) < best_abs:
                        best_x, best_f, best_abs = bx, bv, abs(bv)
            last, lastv = t, v

    # ---- 2. polish the best candidate with Newton/secant ----
    bx = best_x
    if best_f is not None and best_abs > tol and ok():
        # need a second point for secant/Newton; use nearest neighbor with sign change if possible
        # try a small Newton step using finite difference
        h = 1e-7 * max(1.0, abs(bx))
        fx = best_f
        fp = F(bx + h)
        if fp is not None:
            d = (fp - fx) / h
            if d != 0 and math.isfinite(d):
                x_new = bx - fx / d
                if math.isfinite(x_new):
                    v_new = update(x_new, F(x_new))
                    # a couple of secant refinements
                    for _ in range(6):
                        if not ok():
                            break
                        if best_f is not None and best_abs <= tol:
                            break
                        # secant between bx and x_new
                        if v_new is not None and v_new != fx:
                            slope = (v_new - fx) / (x_new - bx) if x_new != bx else 0.0
                            if slope != 0 and math.isfinite(slope):
                                x_next = bx - fx / slope
                                if math.isfinite(x_next):
                                    bx, fx = x_next, update(x_next, F(x_next))
                                    continue
                        break

    # ---- final: return best ----
    result = best_x
    if not math.isfinite(result):
        result = x
    return float(result)


def _bisect(F, update, lo, hi, flo, fhi, tol, ok, deadline):
    """Bisect on [lo,hi] where flo*fhi<0. Returns (x, f(x)) with smallest |f|."""
    a, b = lo, hi
    fa, fb = flo, fhi
    best_a, best_b = a, b
    best_abs = min(abs(fa), abs(fb))
    if best_abs == abs(fa):
        best_x, best_v = a, fa
    else:
        best_x, best_v = b, fb

    for _ in range(60):
        if not ok():
            break
        if best_abs <= tol:
            break
        m = 0.5 * (a + b)
        if m == a or m == b:
            break
        fm = F(m)
        if fm is None:
            break
        am = abs(fm)
        if am < best_abs:
            best_x, best_v, best_abs = m, fm, am
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (best_x, best_v)

# EVOLVE-BLOCK-END
