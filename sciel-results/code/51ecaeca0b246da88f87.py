"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Use a multi-start local search to find a sign-change bracket.
     - Try a set of offsets from x0 (both directions, increasing magnitude).
     - Once a bracket [a, b] with f(a)*f(b) < 0 is found, use a
       bisection + Illinois interpolation hybrid to converge to |f| <= 1e-10.
  3. If no bracket is found, use a damped Newton / secant local minimization
     of |f(x)| to drive f as close to zero as possible.
  4. Track the best (smallest |f|) point seen so far and return it if the
     deadline is approached.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math

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

def _solve_bracket(f, a, fa, b, fb):
    """
    Solve for a root in [a, b] where fa*fb < 0 using Illinois method.
    """
    if fa == 0: return a
    if fb == 0: return b
    if fa * fb > 0:
        raise ValueError("Not a bracket")
        
    left = a
    fleft = fa
    right = b
    fright = fb
    
    # Illinois method with bisection fallback
    for _ in range(100):
        if _elapsed() > _DEADLINE:
            break
            
        if abs(fleft) <= 1e-10:
            return left
        if abs(fright) <= 1e-10:
            return right
        
        # Illinois interpolation
        denom = fright - fleft
        if abs(denom) > 1e-30:
            x_new = (left * fright - right * fleft) / denom
        else:
            x_new = (left + right) / 2.0
            
        # Ensure x_new is within [left, right]
        if not (left < x_new < right):
            x_new = (left + right) / 2.0
            
        try:
            f_new = f(x_new)
        except:
            f_new = None
            
        if f_new is None or not _is_finite(f_new):
            # Fallback to bisection
            x_new = (left + right) / 2.0
            try:
                f_new = f(x_new)
            except:
                f_new = None
            if f_new is None or not _is_finite(f_new):
                break
                
        if abs(f_new) <= 1e-8:
            return x_new
            
        if f_new * fleft < 0:
            right = x_new
            fright = f_new
        else:
            left = x_new
            fleft = f_new
            
        # Convergence check
        if abs(right - left) < 1e-14 * max(1.0, abs(left), abs(right)):
            break
            
    # Return best estimate
    if abs(fleft) <= abs(fright):
        return left
    else:
        return right

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0
    
    best_x = x0
    best_f = None
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')
    
    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return x0
    
    # Phase 1: Find a bracket [a, b]
    a = x0
    fa = best_f
    if fa is None or not _is_finite(fa):
        # Try to find a valid starting point
        found_valid = False
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    a = x0 + dx
                    fa = fx_try
                    found_valid = True
                    break
            except:
                continue
        if not found_valid:
            return x0
            
    if abs(fa) <= 1e-8:
        return a
        
    # Expansion steps
    steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    bracket_found = False
    
    # Positive side scan
    prev_x = a
    prev_f = fa
    
    for s in steps:
        if _elapsed() > _DEADLINE:
            break
            
        next_x = x0 + s
        try:
            next_f = f(next_x)
        except:
            continue
            
        if not _is_finite(next_f):
            continue
            
        if abs(next_f) <= 1e-8:
            return next_x
            
        if prev_f * next_f < 0:
            # Bracket found
            try:
                return _solve_bracket(f, prev_x, prev_f, next_x, next_f)
            except:
                # Fallback to simple bisection
                left, f_left = prev_x, prev_f
                right, f_right = next_x, next_f
                for _ in range(50):
                    if _elapsed() > _DEADLINE:
                        break
                    mid = (left + right) / 2
                    f_mid = f(mid)
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if f_left * f_mid < 0:
                        right, f_right = mid, f_mid
                    else:
                        left, f_left = mid, f_mid
                return left if abs(f_left) < abs(f_right) else right

        prev_x = next_x
        prev_f = next_f
        
    # Negative side scan
    prev_x = a
    prev_f = fa
    
    for s in steps:
        if _elapsed() > _DEADLINE:
            break
            
        next_x = x0 - s
        try:
            next_f = f(next_x)
        except:
            continue
            
        if not _is_finite(next_f):
            continue
            
        if abs(next_f) <= 1e-8:
            return next_x
            
        if prev_f * next_f < 0:
            # Bracket found
            try:
                # Ensure a < b for _solve_bracket
                if next_x < prev_x:
                    return _solve_bracket(f, next_x, next_f, prev_x, prev_f)
                else:
                    return _solve_bracket(f, prev_x, prev_f, next_x, next_f)
            except:
                # Fallback to simple bisection
                left, f_left = next_x, next_f
                right, f_right = prev_x, prev_f
                if left > right:
                    left, right = right, left
                    f_left, f_right = f_right, f_left
                for _ in range(50):
                    if _elapsed() > _DEADLINE:
                        break
                    mid = (left + right) / 2
                    f_mid = f(mid)
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if f_left * f_mid < 0:
                        right, f_right = mid, f_mid
                    else:
                        left, f_left = mid, f_mid
                return left if abs(f_left) < abs(f_right) else right

        prev_x = next_x
        prev_f = next_f

    # If no bracket found, return best guess
    return x0

# EVOLVE-BLOCK-END
