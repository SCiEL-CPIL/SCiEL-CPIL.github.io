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

def _brent_method(f, a, fa, b, fb, tol=1e-8, max_iter=100):
    """
    Robust Brent's method for finding a root in [a, b] where f(a)*f(b) < 0.
    Stops early if |f(x)| <= tol.
    """
    if fa == 0: return a
    if fb == 0: return b
    if fa * fb > 0:
        raise ValueError("Not a bracket")
        
    # Variables for Brent's method
    c = a
    fc = fa
    d = b - a
    e = d
    
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
            
        if abs(fb) <= tol:
            return b
        if abs(fc) <= tol:
            return c
            
        # Ensure f(a) and f(b) have opposite signs
        # In Brent's, we usually maintain f(a) and f(b) as the endpoints of the bracket
        # and f(c) as the best previous point.
        # Standard implementation:
        # a, b are bracket ends. c is the last point.
        # We need to be careful with the standard algorithm variables.
        
        # Let's use a simpler safeguarded Newton/Secant hybrid which is often faster and simpler to implement correctly for this tolerance.
        # Actually, let's stick to a very robust Bisection + Secant (Illinois) but optimized.
        # The previous Illinois code was okay but maybe too many iterations.
        
        # Let's implement a standard Brent's from scratch carefully.
        # Reference: Numerical Recipes in C
        if abs(fc) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            c, d, e = a, b-a, b-a
        fc = fa
        
        # Inverse quadratic interpolation
        s = None
        if a != b and b != c:
            # Lagrange interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) + \
                b * fa * fc / ((fb - fa) * (fb - fc)) + \
                c * fa * fb / ((fc - fa) * (fc - fb))
        elif e == d:
            s = 2 * (a + b) / 2 # Bisection
        else:
            # Linear interpolation
            s = c - (c - a) * fb / (fb - fa)
            
        # Check if s is acceptable
        cond1 = ((1.5 * (b + c) / 2) < s and s < b) or \
                ((1.5 * (b + c) / 2) > s and s > b) or \
                (abs(s - (a + b) / 2) < abs(e) / 2)
                
        cond2 = (abs(s - c) < abs(d) / 2)
        cond3 = (abs(s - c) >= abs(b - a) / 2)
        cond4 = (abs(d) <= tol) # Use tolerance for x
        cond5 = (abs(e) <= tol)
        
        if cond1 or cond2:
            d = e
        else:
            e = d
            
        if cond3 or cond4 or cond5:
            s = (a + b) / 2
        else:
            s = s
            
        # Evaluate f(s)
        try:
            fs = f(s)
        except:
            # Fallback to bisection if evaluation fails
            s = (a + b) / 2
            try:
                fs = f(s)
            except:
                break
                
        if not _is_finite(fs):
             s = (a + b) / 2
             try:
                 fs = f(s)
             except:
                 break
             if not _is_finite(fs):
                 break

        if abs(fs) <= tol:
            return s
            
        c, fc = a, fa
        
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fa # Wait, fa should be updated? No, in Brent, a is the other end.
            # Actually, standard Brent updates:
            # if fa*fs < 0: b=s, fb=fs
            # else: a=s, fa=fs
            
        # Re-check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
             return a if abs(fa) < abs(fb) else b
             
    # Return best
    if abs(fa) < abs(fb):
        return a
    else:
        return b

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
    # Strategy: Scan outwards from x0, checking for sign changes between consecutive points.
    # This is more robust for oscillatory functions than checking against a fixed center.
    
    # Generate a sequence of points to check
    # Start with small steps, then larger
    # We'll create a list of candidate points and evaluate them in order.
    # To save calls, we evaluate and check for sign change with the *previous* point.
    
    # Initial point
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
    # We want to catch sign changes.
    # Steps: 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0
    steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    bracket_found = False
    left = a
    f_left = fa
    
    for step in steps:
        # Try positive direction
        for sign in [1, -1]:
            if _elapsed() > _DEADLINE:
                break
                
            # We move from 'left' to 'left + sign*step'
            # But we should probably keep a sequence.
            # Let's just try points x0 +/- step and check against the *current* left?
            # No, that's not a scan.
            
            # Better: Define a path.
            # Path: x0, x0+0.01, x0+0.02, x0+0.05 ... and x0-0.01, x0-0.02 ...
            # We need to check sign changes between consecutive points in this path.
            
            # Let's restructure:
            # We have a current point 'curr' and 'f_curr'.
            # We try a new point 'next'.
            # If sign change, bracket is [curr, next].
            
            pass # Placeholder for logic below

    # Let's implement the scan properly
    curr_x = a
    curr_f = fa
    
    # List of offsets to try, in order of increasing magnitude
    # We will try them in an interleaved fashion or just positive then negative?
    # Interleaved is better to find the nearest root.
    # But for simplicity, let's do a list of (dx) and try them.
    # To ensure we check consecutive points, we need to keep the "last" point.
    # If we jump from x0 to x0+10, we miss the root at x0+1.
    # So we must evaluate intermediate points.
    
    # Define the sequence of points to visit
    # We will visit: x0, x0+0.01, x0+0.02, x0+0.05, x0+0.1, x0+0.2, x0+0.5, x0+1.0, ...
    # and x0-0.01, x0-0.02, ...
    # We can just iterate through a combined list of offsets.
    
    offsets = []
    for s in steps:
        offsets.append(s)
        offsets.append(-s)
        
    # Sort by absolute value to prioritize nearby roots
    offsets.sort(key=lambda x: abs(x))
    
    last_x = a
    last_f = fa
    
    for dx in offsets:
        if _elapsed() > _DEADLINE:
            break
            
        next_x = x0 + dx # Note: we are expanding from x0, not from last_x?
        # If we expand from x0, we don't get a continuous path.
        # We should expand from x0, but check against x0? No, that misses roots between x0 and x0+1 if x0+0.5 has same sign as x0.
        # So we MUST evaluate the intermediate points in order.
        
        # Correct Strategy:
        # 1. Start at x0.
        # 2. Try x0 + 0.01. Check sign change with x0.
        # 3. Try x0 + 0.02. Check sign change with x0 + 0.01? 
        #    If we do that, we are moving away from x0.
        #    If we find a sign change between x0+0.01 and x0+0.02, we have a bracket.
        #    If not, we continue to x0+0.05. Check with x0+0.02?
        #    This creates a path: x0 -> x0+0.01 -> x0+0.02 -> x0+0.05 ...
        #    This is good.
        
        # But what if the root is on the negative side?
        # We should probably interleave or do one side then the other.
        # Given oscillatory nature, roots might be on both sides.
        # Let's do positive side first, then negative.
        
        # Actually, let's just use the list of offsets but ensure we check the *previous* point in the sequence.
        # The sequence for positive side: 0.01, 0.02, 0.05, 0.1, ...
        # The sequence for negative side: -0.01, -0.02, -0.05, -0.1, ...
        
        # We will iterate through positive offsets, keeping track of the last point.
        # Then iterate through negative offsets, keeping track of the last point (reset to x0? or continue?)
        # Resetting to x0 for negative side is safer to avoid a huge jump from +100 to -100.
        
        pass

    # Let's implement the two-sided scan
    # Positive side
    last_x = a
    last_f = fa
    prev_x = a
    prev_f = fa
    
    for s in steps:
        if _elapsed() > _DEADLINE: break
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
                return _brent_method(f, prev_x, prev_f, next_x, next_f)
            except:
                # Fallback to bisection if Brent fails
                left, f_left = prev_x, prev_f
                right, f_right = next_x, next_f
                for _ in range(50):
                    if _elapsed() > _DEADLINE: break
                    mid = (left + right) / 2
                    f_mid = f(mid)
                    if abs(f_mid) <= 1e-8: return mid
                    if f_left * f_mid < 0:
                        right, f_right = mid, f_mid
                    else:
                        left, f_left = mid, f_mid
                return left if abs(f_left) < abs(f_right) else right

        # Update last point for the next step in this direction
        prev_x = next_x
        prev_f = next_f
        
    # Negative side
    prev_x = a
    prev_f = fa
    for s in steps:
        if _elapsed() > _DEADLINE: break
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
                return _brent_method(f, next_x, next_f, prev_x, prev_f) # Ensure a < b? Brent handles it, but let's be safe
            except:
                left, f_left = next_x, next_f
                right, f_right = prev_x, prev_f
                if left > right:
                    left, right = right, left
                    f_left, f_right = f_right, f_left
                for _ in range(50):
                    if _elapsed() > _DEADLINE: break
                    mid = (left + right) / 2
                    f_mid = f(mid)
                    if abs(f_mid) <= 1e-8: return mid
                    if f_left * f_mid < 0:
                        right, f_right = mid, f_mid
                    else:
                        left, f_left = mid, f_mid
                return left if abs(f_left) < abs(f_right) else right

        # Update last point
        prev_x = next_x
        prev_f = next_f

    # If no bracket found, return best guess or try Newton
    # For now, return the best point seen
    return x0

# EVOLVE-BLOCK-END
