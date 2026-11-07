"""Structural change: multi-scale dense sampling with adaptive bracket refinement."""
import math
import time
from scipy.optimize import brentq

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480  # Use most of the 500-call budget


def _time_left():
    return _DEADLINE - time.time()


def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for an oscillatory function f."""
    counter = [0]
    max_calls = _CALL_BUDGET

    def wf(x):
        counter[0] += 1
        if counter[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        return f(x)

    best_x = float(x0)
    best_abs = float('inf')

    def try_update(x, val):
        nonlocal best_x, best_abs
        if math.isfinite(val):
            a = abs(val)
            if a < best_abs:
                best_x = x
                best_abs = a
                if best_abs <= 1e-8:
                    return True
        return False

    def safe_eval(x):
        """Evaluate wf(x), return (value, ok). ok=False if exception/non-finite."""
        try:
            v = wf(x)
            if not math.isfinite(v):
                return v, False
            return v, True
        except Exception:
            return float('nan'), False

    x0f = float(x0)
    base = max(1.0, abs(x0f))

    # --- Phase 1: Initial evaluation at x0 ---
    fx0, ok0 = safe_eval(x0f)
    if ok0 and try_update(x0f, fx0):
        return best_x

    # --- Phase 2: Multi-scale dense sampling to find a bracket ---
    # For each scale, sample N points symmetrically around x0.
    # Look for sign changes between consecutive samples.
    # Use adaptive N: fewer points at large scales, more at small scales.
    
    # Scale factors: from very small to large
    # n_samples: number of intervals (so n_samples+1 points)
    sampling_plan = [
        # (scale_factor, n_intervals)
        (1e-6, 20),   # 21 points, very fine local
        (1e-5, 20),   # 21 points
        (1e-4, 16),   # 17 points
        (1e-3, 16),   # 17 points
        (1e-2, 12),   # 13 points
        (1e-1, 10),   # 11 points
        (1.0, 8),     # 9 points
        (5.0, 8),     # 9 points
        (10.0, 6),    # 7 points
        (50.0, 6),    # 7 points
        (100.0, 4),   # 5 points
        (500.0, 4),   # 5 points
    ]

    bracket_found = False
    a_br, b_br = None, None
    fa_br, fb_br = None, None

    for scale, n_intervals in sampling_plan:
        if counter[0] >= max_calls or _time_left() < 5.0:
            break
        
        h = scale * base
        # Generate n_intervals+1 points from x0 - h to x0 + h
        # Actually, let's do symmetric: from x0 - n_intervals*h/2 to x0 + n_intervals*h/2
        # This keeps x0 centered
        half_range = (n_intervals / 2.0) * h
        start = x0f - half_range
        step = (2.0 * half_range) / n_intervals
        
        prev_x = None
        prev_v = None
        prev_ok = False
        
        for i in range(n_intervals + 1):
            if counter[0] >= max_calls or _time_left() < 5.0:
                break
            
            x_i = start + i * step
            v_i, ok_i = safe_eval(x_i)
            
            if ok_i and try_update(x_i, v_i):
                return best_x
            
            if ok_i and prev_ok:
                # Check for sign change
                if prev_v == 0.0:
                    return prev_x
                if v_i == 0.0:
                    return x_i
                if prev_v * v_i < 0.0:
                    a_br, b_br = prev_x, x_i
                    fa_br, fb_br = prev_v, v_i
                    bracket_found = True
                    break
            
            prev_x = x_i
            prev_v = v_i
            prev_ok = ok_i
        
        if bracket_found:
            break

    # --- Phase 3: Brent's method on the bracket ---
    if bracket_found:
        try:
            res = brentq(wf, a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=60)
            if math.isfinite(res):
                fr, okr = safe_eval(res)
                if okr and try_update(res, fr):
                    return best_x
                if okr and abs(fr) < best_abs:
                    best_x = res
                    best_abs = abs(fr)
        except Exception:
            pass

    # --- Phase 4: If no bracket found, try a finer local search around best_x ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 3.0:
        # Try very fine sampling around the best point found so far
        for scale in [1e-7, 1e-8, 1e-9]:
            if counter[0] >= max_calls or _time_left() < 3.0:
                break
            h = scale * max(1.0, abs(best_x))
            n_pts = 11  # 11 points, 10 intervals
            start = best_x - (n_pts // 2) * h
            for i in range(n_pts):
                if counter[0] >= max_calls or _time_left() < 3.0:
                    break
                x_i = start + i * h
                v_i, ok_i = safe_eval(x_i)
                if ok_i and try_update(x_i, v_i):
                    return best_x

    return best_x
