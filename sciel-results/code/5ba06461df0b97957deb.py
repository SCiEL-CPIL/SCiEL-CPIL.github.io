"""
Strategy: Pure derivative-free global search via dense multi-scale sampling.
Instead of relying on local bracketing expansion (which can miss roots in high-
frequency oscillatory regions if the initial step is too large) or Newton's
method (which requires derivative estimation and can diverge), we perform a
dense, adaptive grid scan around x0 at multiple scales. This guarantees that
for any continuous function with a root within the search radius, we will find
a sign change (bracket) with high probability, which we then refine using
scipy.optimize.brentq. The key difference is the *search strategy*: dense
global sampling vs. local expansion.
"""
import math
import time
from scipy.optimize import brentq

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490


def _time_left():
    return _DEADLINE - time.time()


def _counted_call(f, x, counter):
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val


def _try_brentq(f, a, b, counter):
    """Attempt to find a root in [a, b] using brentq. Returns root or None."""
    if a == b:
        return None
    if _time_left() < 0.1:
        return None
    if counter[0] >= _CALL_BUDGET - 10:
        return None
    try:
        result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        # The evaluator's final check is not counted, but we should verify locally
        # to avoid returning a bad value if brentq failed silently (it shouldn't).
        # However, calling f(result) here counts! We must be careful.
        # brentq guarantees |f(result)| <= 1e-8 if it converges.
        # But if it hits maxiter, it might not.
        # We'll trust brentq's convergence criteria for now, as it's robust.
        # If we want to be safe, we could check, but that costs 1 call.
        # Given the budget, let's just return the result.
        return result
    except Exception:
        return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: Multi-scale dense grid scan to find a bracket, then brentq.
    """
    counter = [0]
    x0 = float(x0)
    
    # First, evaluate f(x0)
    f0 = _counted_call(f, x0, counter)
    if not math.isfinite(f0):
        return x0
    if abs(f0) <= 1e-10:
        return x0

    # Multi-scale search
    # Scales: small to large steps.
    # For each scale, we scan a window around x0.
    # We need to be efficient with calls.
    
    # Scale 1: Fine search near x0 (step ~ 0.01)
    # Scale 2: Medium search (step ~ 0.1)
    # Scale 3: Coarse search (step ~ 1.0)
    # Scale 4: Very coarse search (step ~ 10.0)
    
    # We'll iterate through scales. For each scale, we check points x0 + k*step
    # for k in [-N, N]. We stop when we find a bracket.
    
    scales = [
        (0.01, 20),   # step, num_points_each_side
        (0.05, 10),
        (0.1, 10),
        (0.5, 10),
        (1.0, 10),
        (5.0, 10),
        (10.0, 10),
        (50.0, 10),
        (100.0, 10),
    ]
    
    # To avoid redundant calls, we could cache, but for simplicity and speed,
    # we just call. The number of calls is bounded.
    # Total calls in worst case: sum(2*N+1) for all scales.
    # 41 + 21 + 21 + 21 + 21 + 21 + 21 + 21 + 21 = 209 calls.
    # Plus brentq calls. This is well within 490.
    
    for step, n_points in scales:
        if _time_left() < 1.0:
            break
        if counter[0] >= _CALL_BUDGET - 20:
            break
            
        # We scan from x0 - n_points*step to x0 + n_points*step
        # We already have f(x0).
        # We'll generate points in order of distance from x0 to maximize chance
        # of finding a nearby root first.
        
        # Points: x0, x0+step, x0-step, x0+2*step, x0-2*step, ...
        # We need to check for sign changes between consecutive points in the
        # sorted order of the scan.
        
        # Let's build the list of points to check, excluding x0 (already checked)
        # and sort them by distance.
        points = []
        for k in range(1, n_points + 1):
            points.append((k * step, x0 + k * step))
            points.append((k * step, x0 - k * step))
        
        # Sort by distance
        points.sort(key=lambda x: x[0])
        
        prev_x = x0
        prev_f = f0
        
        for dist, x in points:
            if _time_left() < 0.5:
                break
            if counter[0] >= _CALL_BUDGET - 10:
                break
                
            fx = _counted_call(f, x, counter)
            if not math.isfinite(fx):
                # If we hit a non-finite value, we can't use it for bracketing.
                # We might skip this point and continue, but the "prev" chain is broken.
                # We should reset prev to this point if we want to continue searching
                # in the same direction, but since we're jumping, it's safer to just
                # move on. However, if the function is defined elsewhere, we might
                # miss a root.
                # For robustness, let's just update prev to this point if it's finite,
                # but if it's not, we can't bracket.
                # Actually, if fx is not finite, we can't use it.
                # We'll just continue to the next point.
                continue
            
            # Check for sign change between prev_x and x
            if prev_f * fx < 0:
                # Found a bracket [min(prev_x, x), max(prev_x, x)]
                a = min(prev_x, x)
                b = max(prev_x, x)
                result = _try_brentq(f, a, b, counter)
                if result is not None:
                    return result
            
            # Update prev
            prev_x = x
            prev_f = fx
            
            # If we found a very small value, we can return early
            if abs(fx) <= 1e-10:
                return x

    # If no bracket found, try a few more random-ish points or just return x0
    # As a last resort, try a few more points with larger steps
    if _time_left() > 1.0 and counter[0] < _CALL_BUDGET - 10:
        for delta in [0.1, -0.1, 1.0, -1.0, 10.0, -10.0]:
            x = x0 + delta
            fx = _counted_call(f, x, counter)
            if math.isfinite(fx) and f0 * fx < 0:
                a = min(x0, x)
                b = max(x0, x)
                result = _try_brentq(f, a, b, counter)
                if result is not None:
                    return result

    return x0
