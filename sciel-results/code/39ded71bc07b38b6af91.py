"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Systematic bracket search: try small steps first (root may be near x0),
     then expand geometrically. For oscillatory functions, try multiple
     offset patterns to catch sign changes.
  3. Use scipy.optimize.brentq if bracket found.
  4. Fallback: secant/Newton iteration from best point, then bounded
     minimization of |f(x)|.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _update_best(best_x, best_f, x, fx):
    """Update best if fx is finite and has smaller absolute value."""
    if _is_finite(fx) and abs(fx) < abs(best_f):
        return x, fx
    return best_x, best_f

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Robust bracket search with geometric expansion.
    3. Use scipy.optimize.brentq if bracket found.
    4. Fallback to scipy.optimize.minimize_scalar on |f(x)| if no bracket.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    best_x = x0
    best_f = float('inf')

    try:
        f0 = f(x0)
        if _is_finite(f0):
            best_f = f0
            if abs(f0) <= 1e-8:
                return x0
    except:
        pass

    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) < 0
    # We try to expand an interval around x0
    # Start with a small step, increase geometrically
    
    a = x0
    fa = best_f
    if not _is_finite(fa):
        # If x0 is bad, try to find a good point
        found_good = False
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            try:
                x_try = x0 + dx
                f_try = f(x_try)
                if _is_finite(f_try):
                    a = x_try
                    fa = f_try
                    found_good = True
                    if abs(f_try) <= 1e-8:
                        return x_try
                    break
            except:
                continue
        if not found_good:
            return x0

    b = None
    fb = None
    
    # Geometric expansion
    # Try steps: 1, 2, 4, 8, ... in both directions
    # Also try smaller initial steps if x0 is large
    base_step = max(1.0, abs(x0) * 0.1)
    
    for i in range(20):
        if _elapsed() > _DEADLINE - 5:
            break
            
        step = base_step * (2 ** i)
        
        # Check right
        try:
            b_try = a + step
            fb_try = f(b_try)
            if _is_finite(fb_try):
                best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            pass
        
        # Check left
        try:
            b_try = a - step
            fb_try = f(b_try)
            if _is_finite(fb_try):
                best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            pass

    # Phase 2: If bracket found, use brentq
    if b is not None and fb is not None and fa * fb < 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if _is_finite(root):
                return root
        except Exception:
            pass

    # Phase 3: Fallback - Minimize |f(x)|
    # Try multiple starting points for local minimization
    # This helps if the root is a tangent (no sign change) or if bracket search missed it
    
    def abs_f(x):
        try:
            val = f(x)
            if not _is_finite(val):
                return 1e100
            return abs(val)
        except:
            return 1e100

    # Try minimization from x0 and nearby points
    candidates = [x0, x0 + 1.0, x0 - 1.0, x0 + 0.5, x0 - 0.5]
    
    best_candidate_x = x0
    best_candidate_val = float('inf')

    for start in candidates:
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Bounded minimization around start
        # Use a window that grows if needed, but keep it reasonable for speed
        # We'll try a few window sizes
        for width in [1.0, 10.0, 100.0]:
            if _elapsed() > _DEADLINE - 5:
                break
            
            lower = start - width
            upper = start + width
            
            try:
                res = minimize_scalar(abs_f, bounds=(lower, upper), method='bounded', options={'xatol': 1e-12})
                if res.success and _is_finite(res.x):
                    if res.fun < best_candidate_val:
                        best_candidate_val = res.fun
                        best_candidate_x = res.x
                        if best_candidate_val <= 1e-8:
                            return best_candidate_x
            except Exception:
                pass

    # If we found a good candidate, return it
    if best_candidate_val <= 1e-8:
        return best_candidate_x

    # If all else fails, return the best point we've seen
    # We should probably evaluate f(best_x) to be sure, but we might not have a better one
    # The evaluator checks |f(x)| <= 1e-8. If we can't achieve that, we fail this equation.
    # Returning the best effort is the standard protocol.
    
    # One last desperate check: try a few random points? No, deterministic is better for reproducibility in eval.
    # Let's just return the best_x we tracked or x0.
    # Actually, let's try to minimize again with a very wide bound if we have time?
    # No, bounded minimization is safer.
    
    return best_candidate_x if best_candidate_val < best_f else x0

# EVOLVE-BLOCK-END
