"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Secant method (reuses previous evaluations, 1 call per iteration after first 2)
    2. Newton's method with finite-difference derivative as fallback
    3. Bisection with bracket expansion for guaranteed convergence
    
    Returns the best x found (minimizing |f(x)|) within the call budget.
    """
    tol = 1e-8
    max_calls = 480  # stay well under 500
    
    # Count function calls
    call_count = [0]
    
    def counted_eval(x):
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        val = f(x)
        if not _is_finite(val):
            raise ValueError("non-finite value")
        return val
    
    best_x = float(x0)
    best_fval = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fval
        if _is_finite(x) and abs(fx) < best_fval:
            best_x = x
            best_fval = abs(fx)
    
    try:
        # Phase 1: Secant method
        x = float(x0)
        h = 1e-4 * max(1.0, abs(x))
        x1 = x + h
        fx1 = counted_eval(x1)
        update_best(x1, fx1)
        
        if abs(fx1) <= tol:
            return x1
        
        x0_prev = x
        fx0 = counted_eval(x0_prev)
        update_best(x0_prev, fx0)
        
        if abs(fx0) <= tol:
            return x0_prev
        
        # Secant iterations
        for i in range(1, 200):
            if _time_left() < 2.0:
                break
            if fx1 == fx0:
                break
            x_new = x1 - fx1 * (x1 - x0_prev) / (fx1 - fx0)
            if not _is_finite(x_new):
                break
            fx_new = counted_eval(x_new)
            update_best(x_new, fx_new)
            if abs(fx_new) <= tol:
                return x_new
            x0_prev, fx0 = x1, fx1
            x1, fx1 = x_new, fx_new
        
        # Check if we're good enough
        if best_fval <= tol:
            return best_x
        
        # Phase 2: Try Newton's method from best point
        x = best_x
        for i in range(50):
            if _time_left() < 2.0:
                break
            if call_count[0] >= max_calls - 5:
                break
            fx = counted_eval(x)
            update_best(x, fx)
            if abs(fx) <= tol:
                return x
            h = 1e-6 * max(1.0, abs(x))
            fx_h = counted_eval(x + h)
            d = (fx_h - fx) / h
            if d == 0 or not _is_finite(d):
                break
            x_new = x - fx / d
            if not _is_finite(x_new):
                break
            x = x_new
        
        # Phase 3: Bisection if we haven't converged
        if best_fval > tol and _time_left() > 2.0 and call_count[0] < max_calls - 10:
            # Try to find a bracket
            x_lo = best_x - 1.0
            x_hi = best_x + 1.0
            fx_lo = None
            fx_hi = None
            try:
                fx_lo = counted_eval(x_lo)
                update_best(x_lo, fx_lo)
                fx_hi = counted_eval(x_hi)
                update_best(x_hi, fx_hi)
            except (RuntimeError, ValueError):
                pass
            
            if fx_lo is not None and fx_hi is not None:
                # Expand bracket if needed
                scale = 1.0
                for _ in range(20):
                    if _time_left() < 2.0 or call_count[0] >= max_calls - 5:
                        break
                    if fx_lo * fx_hi < 0:
                        break
                    x_lo = best_x - scale
                    x_hi = best_x + scale
                    try:
                        fx_lo = counted_eval(x_lo)
                        update_best(x_lo, fx_lo)
                        fx_hi = counted_eval(x_hi)
                        update_best(x_hi, fx_hi)
                    except (RuntimeError, ValueError):
                        break
                    scale *= 2.0
                
                if fx_lo * fx_hi < 0:
                    # Bisection
                    for _ in range(60):
                        if _time_left() < 2.0 or call_count[0] >= max_calls - 1:
                            break
                        x_mid = 0.5 * (x_lo + x_hi)
                        fx_mid = counted_eval(x_mid)
                        update_best(x_mid, fx_mid)
                        if abs(fx_mid) <= tol:
                            return x_mid
                        if fx_lo * fx_mid < 0:
                            x_hi = x_mid
                            fx_hi = fx_mid
                        else:
                            x_lo = x_mid
                            fx_lo = fx_mid
                        if abs(x_hi - x_lo) < 1e-12:
                            break
        
        return best_x
    
    except (RuntimeError, ValueError, OverflowError):
        # Return best x found so far
        if best_fval < float('inf') and _is_finite(best_x):
            return best_x
        return float(x0)
