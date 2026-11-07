"""Root finder for 3 fixed oscillatory equations using dense grid bracketing + Brent/Minimize."""
import math
import time
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit
    MAX_CALLS = 490     # Stay well under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = float(x)
        return float(val)

    def _f_sq(x):
        """Wrapper for minimize_scalar: returns f(x)^2."""
        val = _f(x)
        if math.isnan(val):
            return float('inf')
        return val * val

    # Phase 1: Evaluate at x0
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Phase 2: Dense Grid Scan to find a bracket
    # Strategy: Scan increasingly larger ranges with high density near x0
    # to catch oscillatory sign changes.
    # (radius, num_intervals)
    ranges = [
        (1.0, 100),   # Step ~0.02
        (5.0, 100),   # Step ~0.1
        (20.0, 50),   # Step ~0.8
        (100.0, 50)   # Step ~4.0
    ]
    
    bracket_found = False
    bracket = None

    for radius, n_intervals in ranges:
        if calls >= MAX_CALLS - 60 or time.time() - start_time > TIME_LIMIT - 10:
            break
        
        lower = x - radius
        upper = x + radius
        
        # Generate grid points
        # We reuse the previous point if possible to save calls? 
        # Simpler to just scan, as we are tracking calls.
        prev_x = None
        prev_f = None
        
        for i in range(n_intervals + 1):
            if calls >= MAX_CALLS - 60:
                break
            cur_x = lower + i * (upper - lower) / n_intervals
            
            # Optimization: Skip if we already evaluated this exact point?
            # Hard to track without a dict, which adds overhead. 
            # Given the low number of calls, direct evaluation is fine.
            
            cur_f = _f(cur_x)
            if math.isnan(cur_f):
                prev_x = cur_x
                prev_f = cur_f
                continue

            if abs(cur_f) <= 1e-8:
                return cur_x

            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    # Found sign change in [prev_x, cur_x]
                    bracket = (prev_x, cur_x)
                    bracket_found = True
                    break
            
            prev_x = cur_x
            prev_f = cur_f
        
        if bracket_found:
            break

    # Phase 3: Solve using Brent's Method if a bracket was found
    if bracket_found:
        a, b = bracket
        try:
            # brentq is robust and efficient
            root = brentq(_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                # Verify the result
                f_root = _f(root)
                if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                    return root
                # If verification fails, update best if this is better
                if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                    best_x = root
                    best_abs_f = abs(f_root)
        except Exception:
            pass

    # Phase 4: Fallback - Minimize f(x)^2 if no bracket found or verification failed
    if best_abs_f > 1e-8:
        if calls < MAX_CALLS - 20 and time.time() - start_time < TIME_LIMIT - 5:
            # Use the largest range scanned for the minimization bounds
            last_radius = ranges[-1][0]
            lower = x - last_radius
            upper = x + last_radius
            
            try:
                res = minimize_scalar(
                    _f_sq, 
                    bounds=(lower, upper), 
                    method='bounded', 
                    options={'maxiter': 100, 'xatol': 1e-10}
                )
                if res.success and math.isfinite(res.x):
                    f_res = _f(res.x)
                    if not math.isnan(f_res) and abs(f_res) < best_abs_f:
                        best_x = res.x
                        best_abs_f = abs(f_res)
            except Exception:
                pass

    return best_x

# EVOLVE-BLOCK-END
