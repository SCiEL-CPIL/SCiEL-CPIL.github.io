"""
Strategy: Global Periodic Scan with Brent's Method.
1. The current approach relies on local expansion from x0, which can be inefficient or fail
   for oscillatory functions if the root is far away or if the expansion step is poorly tuned.
2. This new strategy assumes the functions are likely periodic or bounded in a reasonable range
   (e.g., [-10, 10] or [-100, 100]).
3. It performs a coarse global scan over a wide interval to find *any* sign change (bracket).
   This is robust because it doesn't depend on the local behavior at x0.
4. Once a bracket is found, it uses scipy.optimize.brentq to converge to the root.
5. If no bracket is found in the global scan, it falls back to a finer local scan around x0.
6. We optimize the scan resolution to balance the probability of finding a bracket with
   the cost of the scan.
"""
import math
import time
import numpy as np
from scipy.optimize import brentq

# Global deadline: evaluator kills after 150s wall-clock.
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490  # stay under 500


def _time_left():
    return _DEADLINE - time.time()


def _counted_call(f, x, counter):
    """Call f(x), track count, guard against exceptions / non-finite."""
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val


def _find_bracket_global_scan(f, x0, counter):
    """
    Perform a coarse global scan to find a sign change.
    We scan over a range of [-R, R] where R is chosen to be large enough
    to likely contain a root, but small enough to keep the call count low.
    """
    # Try a few different ranges and resolutions
    # For oscillatory functions, a root is likely within a few periods.
    # Let's try scanning [-10, 10] with 100 points, then [-50, 50] with 100 points, etc.
    
    ranges = [(10, 100), (50, 150), (100, 200), (500, 300)]
    
    for R, n_points in ranges:
        if counter[0] >= _CALL_BUDGET - 20 or _time_left() < 1.0:
            break
        
        # Generate scan points
        # We want to include x0 in the scan if possible, but a global scan is fine.
        # Let's just do a uniform scan.
        xs = np.linspace(-R, R, n_points)
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
                
            fx = _counted_call(f, x, counter)
            
            if math.isfinite(fx):
                if abs(fx) <= 1e-8:
                    return (x, x)
                
                if prev_f is not None and math.isfinite(prev_f):
                    if prev_f * fx < 0:
                        return (prev_x, x)
                
                prev_x = x
                prev_f = fx
            else:
                prev_x = None
                prev_f = None
                
    return None


def _find_bracket_local_scan(f, x0, counter):
    """
    Fallback: Local scan around x0 with adaptive step size.
    """
    scales = [1.0, 10.0, 100.0]
    steps = [20, 50, 100]
    
    for scale, n_points in zip(scales, steps):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        
        a = x0 - scale
        b = x0 + scale
        xs = np.linspace(a, b, n_points)
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
                
            fx = _counted_call(f, x, counter)
            
            if math.isfinite(fx):
                if abs(fx) <= 1e-8:
                    return (x, x)
                
                if prev_f is not None and math.isfinite(prev_f):
                    if prev_f * fx < 0:
                        return (prev_x, x)
                
                prev_x = x
                prev_f = fx
            else:
                prev_x = None
                prev_f = None
                
    return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Try global scan first (robust for oscillatory functions)
    bracket = _find_bracket_global_scan(f, x0, counter)
    
    # 2. If global scan fails, try local scan
    if bracket is None:
        bracket = _find_bracket_local_scan(f, x0, counter)
    
    if bracket is not None:
        a, b = bracket
        if a == b:
            return a
        try:
            # Use brentq with tight tolerances
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            # Verify
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
            
    # 3. Last resort: return x0 (will likely fail, but avoids exception)
    return x0
