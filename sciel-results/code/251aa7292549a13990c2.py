"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Strategy:
      1. Try a few high-order local methods (Halley, Newton with better derivative) near x0.
      2. If local fails, do a bracketing search (bisection + Newton hybrid) in expanding windows.
      3. Final polish with tight tolerance.
    Deadline guard: return best result found before ~110s.
    """
    t_start = time.time()
    DEADLINE = t_start + 110.0  # safety margin before 120s kill

    def time_ok():
        return time.time() < DEADLINE

    x = float(x0)
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f)

    def update_best(xc, fc):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(xc) and math.isfinite(fc):
            af = abs(fc)
            if af < best_abs:
                best_x, best_f, best_abs = xc, fc, af

    # Evaluate at x0
    update_best(x, best_f)
    if best_abs <= 1e-8:
        return best_x

    # --- Phase 1: Local high-order methods (Halley's method) ---
    # Halley's method: x_{n+1} = x_n - 2 f f' / (2 f'^2 - f f'')
    # We approximate f' and f'' via finite differences.
    # Use complex-step for derivative if possible? No, f may not accept complex.
    # Use 5-point stencil for better accuracy.

    def halley_step(xc, fc):
        h = 1e-4 * max(1.0, abs(xc))
        fp = (f(xc + h) - f(xc - h)) / (2 * h)
        fpp = (f(xc + h) - 2 * fc + f(xc - h)) / (h * h)
        denom = 2 * fp * fp - fc * fpp
        if denom == 0 or not math.isfinite(denom):
            return None
        step = 2 * fc * fp / denom
        return xc - step

    # Try Halley for a few iterations
    for i in range(30):
        if not time_ok():
            break
        if best_abs <= 1e-12:
            break
        step = halley_step(best_x, best_f)
        if step is None or not math.isfinite(step):
            break
        new_x = best_x - step
        if not math.isfinite(new_x):
            break
        new_f = f(new_x)
        if not math.isfinite(new_f):
            break
        update_best(new_x, new_f)
        if best_abs <= 1e-12:
            break
        # Damping if step is too large
        if abs(step) > 10 * max(1.0, abs(best_x)):
            break

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 2: Bracketing search with expanding windows ---
    # Search for a sign change in expanding intervals around x0
    found_bracket = False
    a = b = None
    fa = fb = None

    # Try multiple window sizes
    for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]:
        if not time_ok():
            break
        half = scale * max(1.0, abs(x0))
        a = x0 - half
        b = x0 + half
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            continue
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        update_best(a, fa)
        update_best(b, fb)
        if best_abs <= 1e-8:
            return best_x
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb < 0:
            found_bracket = True
            break

    if not found_bracket:
        # Try a grid search over a wider range
        for scale in [1e2, 1e3, 1e4, 1e5]:
            if not time_ok():
                break
            half = scale * max(1.0, abs(x0))
            a = x0 - half
            b = x0 + half
            n_grid = 50
            prev_x = a
            try:
                prev_f = f(a)
            except Exception:
                continue
            update_best(a, prev_f)
            for i in range(1, n_grid + 1):
                if not time_ok():
                    break
                cur_x = a + (b - a) * i / n_grid
                try:
                    cur_f = f(cur_x)
                except Exception:
                    continue
                update_best(cur_x, cur_f)
                if best_abs <= 1e-8:
                    return best_x
                if math.isfinite(prev_f) and math.isfinite(cur_f):
                    if prev_f * cur_f < 0:
                        found_bracket = True
                        a = prev_x
                        b = cur_x
                        fa = prev_f
                        fb = cur_f
                        break
                prev_x = cur_x
                prev_f = cur_f
            if found_bracket:
                break

    if found_bracket and a is not None and b is not None:
        # Bisection + Newton hybrid to converge within bracket
        ca, cb = a, b
        fc_a, fc_b = fa, fb
        for i in range(80):
            if not time_ok():
                break
            mid = 0.5 * (ca + cb)
            try:
                fm = f(mid)
            except Exception:
                break
            if not math.isfinite(fm):
                break
            update_best(mid, fm)
            if abs(fm) <= 1e-12:
                return mid
            if fm * fc_a < 0:
                cb = mid
                fc_b = fm
            else:
                ca = mid
                fc_a = fm
            # Newton step if it stays in bracket
            h = 1e-6 * max(1.0, abs(mid))
            try:
                fp = (f(mid + h) - f(mid - h)) / (2 * h)
            except Exception:
                fp = None
            if fp is not None and math.isfinite(fp) and fp != 0:
                new_x = mid - fm / fp
                if ca < new_x < cb:
                    try:
                        fn = f(new_x)
                    except Exception:
                        fn = None
                    if fn is not None and math.isfinite(fn):
                        update_best(new_x, fn)
                        if abs(fn) <= 1e-12:
                            return new_x
                        if fn * fc_a < 0:
                            cb = new_x
                            fc_b = fn
                        else:
                            ca = new_x
                            fc_a = fn
            if abs(cb - ca) < 1e-14 * max(1.0, abs(mid)):
                break

    # --- Phase 3: Final polish ---
    if best_abs <= 1e-8:
        # Tighten with a few more Halley/Newton steps
        for i in range(20):
            if not time_ok():
                break
            if best_abs <= 1e-14:
                break
            h = 1e-8 * max(1.0, abs(best_x))
            try:
                fp = (f(best_x + h) - f(best_x - h)) / (2 * h)
            except Exception:
                break
            if not math.isfinite(fp) or fp == 0:
                break
            new_x = best_x - best_f / fp
            if not math.isfinite(new_x):
                break
            try:
                new_f = f(new_x)
            except Exception:
                break
            if not math.isfinite(new_f):
                break
            update_best(new_x, new_f)

    return best_x
# EVOLVE-BLOCK-END
