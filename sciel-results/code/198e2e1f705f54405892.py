"""Efficient root finder for oscillatory equations with adaptive multi-scale search."""
import math
import time
from scipy.optimize import brentq

_DEADLINE = time.time() + 135.0
_CALL_BUDGET = 470  # Stay under 500 with safety margin


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

    # --- Phase 2: Multi-scale sampling to find a bracket ---
    # Use a hierarchical approach: start with coarse scales, refine if no bracket found.
    # For oscillatory functions, we need to sample at scales comparable to the period.
    
    # Sampling plan: (scale_factor, n_intervals)
    # We sample symmetrically around x0 with step = scale * base / n_intervals
    # Total range covered: 2 * scale * base
    sampling_plan = [
        # Fine local search first (most likely to find nearby root)
        (1e-4, 40),    # 41 points, range ±1e-4*base
        (1e-3, 40),    # 41 points, range ±1e-3*base
        (1e-2, 30),    # 31 points, range ±1e-2*base
        (1e-1, 24),    # 25 points, range ±1e-1*base
        (1.0, 20),     # 21 points, range ±base
        (5.0, 16),     # 17 points, range ±5*base
        (10.0, 12),    # 13 points, range ±10*base
        (50.0, 10),    # 11 points, range ±50*base
        (100.0, 8),    # 9 points, range ±100*base
        (500.0, 6),    # 7 points, range ±500*base
        (1000.0, 5),   # 6 points, range ±1000*base
    ]

    bracket_found = False
    a_br, b_br = None, None
    fa_br, fb_br = None, None

    for scale, n_intervals in sampling_plan:
        if counter[0] >= max_calls or _time_left() < 5.0:
            break
        
        h = scale * base
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

    # --- Phase 3: If no bracket found, try one-sided expansion ---
    if not bracket_found and counter[0] < max_calls - 20 and _time_left() > 5.0:
        # Try expanding outward from x0 in both directions
        for direction in [1, -1]:
            if counter[0] >= max_calls - 10 or _time_left() < 5.0:
                break
            
            prev_x = x0f
            prev_v, prev_ok = safe_eval(x0f)
            
            for k in range(1, 50):
                if counter[0] >= max_calls - 10 or _time_left() < 5.0:
                    break
                
                x_i = x0f + direction * k * base * 0.5
                v_i, ok_i = safe_eval(x_i)
                
                if ok_i and try_update(x_i, v_i):
                    return best_x
                
                if ok_i and prev_ok and prev_v * v_i < 0.0:
                    a_br, b_br = prev_x, x_i
                    fa_br, fb_br = prev_v, v_i
                    bracket_found = True
                    break
                
                if ok_i:
                    prev_x = x_i
                    prev_v = v_i
                    prev_ok = True
            
            if bracket_found:
                break

    # --- Phase 4: Brent's method on the bracket ---
    if bracket_found:
        try:
            res = brentq(wf, a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=50)
            if math.isfinite(res):
                fr, okr = safe_eval(res)
                if okr and try_update(res, fr):
                    return best_x
                if okr and abs(fr) < best_abs:
                    best_x = res
                    best_abs = abs(fr)
        except Exception:
            pass

    # --- Phase 5: Fine local search around best_x if not converged ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 3.0:
        for scale in [1e-5, 1e-6, 1e-7]:
            if counter[0] >= max_calls or _time_left() < 3.0:
                break
            h = scale * max(1.0, abs(best_x))
            n_pts = 15
            start = best_x - (n_pts // 2) * h
            for i in range(n_pts):
                if counter[0] >= max_calls or _time_left() < 3.0:
                    break
                x_i = start + i * h
                v_i, ok_i = safe_eval(x_i)
                if ok_i and try_update(x_i, v_i):
                    return best_x

    return best_x
