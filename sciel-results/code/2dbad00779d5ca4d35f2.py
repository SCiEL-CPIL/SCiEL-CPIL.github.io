"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ZeroDivisionError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val

    def is_good(x, fx):
        if fx is None:
            return False
        return abs(fx) <= TOL

    def siqi(x0, max_iter=40):
        """
        Safeguarded Inverse Quadratic Interpolation (SIQI)
        Uses 3 points to fit an inverse quadratic and finds the root.
        Safeguarded by bisection if the interpolation step is too large or goes out of bounds.
        """
        nonlocal evals
        
        # Initial 3 points
        # We need to establish a bracket or just 3 points. 
        # For ill-conditioned, we might not have a bracket. 
        # SIQI usually works best with a bracket, but can be used as a 
        # derivative-free method if we just track the best point and 
        # use interpolation to guess the next one.
        
        # Strategy: 
        # 1. Start with x0.
        # 2. Generate x1, x2 nearby.
        # 3. If we have a sign change, use standard SIQI with bracketing.
        # 4. If not, use a "minimization" style SIQI where we interpolate 
        #    to find where f crosses 0, but if it doesn't, we fall back to 
        #    exploring a wider range or using the best point found.
        
        # Let's stick to a robust version:
        # If we can find a bracket [a, b] where f(a)*f(b) < 0, we use SIQI to converge.
        # If we can't, we use a secant-like approach on the best 3 points, 
        # but with safeguards to prevent divergence.
        
        # Actually, a simpler and very robust derivative-free method is 
        # just the Secant method with a "best point" memory, but SIQI is 
        # specifically requested.
        
        # Let's implement a standard SIQI that requires a bracket.
        # If no bracket is found quickly, it will return the best point found.
        
        # First, try to find a bracket starting from x0
        a = float(x0)
        fa = call_f(a)
        if fa is None:
            return a, None
        if is_good(a, fa):
            return a, fa
            
        # Expand to find a sign change
        b = a
        fb = None
        step = 1e-2 * max(1.0, abs(a))
        
        for _ in range(10):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                break
            b = a + step
            fb = call_f(b)
            if fb is None:
                break
            if is_good(b, fb):
                return b, fb
            if fa * fb < 0:
                # Bracket found [a, b]
                break
            # Try the other side if no sign change
            b = a - step
            fb = call_f(b)
            if fb is None:
                break
            if is_good(b, fb):
                return b, fb
            if fa * fb < 0:
                # Bracket found [b, a]
                # Swap so a < b
                a, b = b, a
                fa, fb = fb, fa
                break
            step *= 2
            
        if fa is None or fb is None or fa * fb > 0:
            # No bracket found. Return best.
            return best_x, best_f

        # Now we have a bracket [a, b] with fa*fb < 0
        # Initialize SIQI variables
        # c is the point with the smaller |f|
        if abs(fa) < abs(fb):
            c = a
            fc = fa
        else:
            c = b
            fc = fb
            
        d = b - a
        e = d
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            
            # Invariant: fa * fb < 0
            if fa * fb > 0:
                a = c
                fa = fc
                d = b - a
                e = d
                
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
                
            tol1 = 2 * 2.22e-16 * abs(b)
            delta = 0.5 * (c - a)
            
            if abs(delta) <= tol1 or fb == 0:
                return b, fb
            
            if abs(e) < tol1 or abs(fa) <= abs(fb):
                # Bisection
                e = delta
                d = fb / fa
            else:
                # Inverse Quadratic Interpolation
                S = fa / fc
                T1 = fb / fc
                T2 = fb / fa
                d = T1 * (T1 - S) * (b - c) / (fa - fc)
                S = T1 * T2 * (c - a) / (fa - fc)
                
                # Check if d is in the valid range
                if ((d > 0.5 * delta * (c - a)) or (d < 0.5 * delta * (b - c))):
                    d = delta
                if ((c - b) if (d >= 0) else (b - c)) < (3 * tol1 / 2):
                    d = delta * ((c - b) if (fb >= 0) else (b - c)) / abs(fb)
                    if d < 0:
                        d = -d
                    else:
                        d = d
                    if abs(d) < tol1:
                        d = delta
            
            e = d
            b_new = b + d
            fb_new = call_f(b_new)
            if fb_new is None:
                break
            
            if is_good(b_new, fb_new):
                return b_new, fb_new
            
            # Update bracket
            if fa * fb_new > 0:
                a = c
                fa = fc
                d = b_new - a
                e = d
            else:
                c = b
                fc = fb
                
            b = b_new
            fb = fb_new
            
        return b, fb

    # Strategy: Multi-start stochastic search
    # 1. Evaluate x0
    # 2. Generate a set of random restart points around x0
    # 3. For each point, run SIQI
    # 4. Keep track of the best result
    
    # First, try the given x0 with SIQI
    x_res, f_res = siqi(x0)
    if is_good(x_res, f_res):
        return x_res

    # If x0 failed, try random restarts
    # We need to be careful with the budget. 
    # Let's use a fixed number of restarts, but stop early if we succeed or run out of time/evals.
    
    # Scale of exploration
    base_scale = max(1.0, abs(x0))
    
    # Define a set of scales for the random offsets
    # Log-uniform distribution of scales
    scales = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # We will iterate through scales and try a few random points at each scale
    # This ensures we look at local, medium, and global neighborhoods.
    
    # To be efficient, we'll use a deterministic pseudo-random sequence based on a seed 
    # derived from x0 to ensure reproducibility if needed, or just use random.
    # Using random is fine since we don't need reproducibility across runs, 
    # but it's good to be stable within a run.
    
    rng = random.Random(42) # Fixed seed for consistency? Or just random? 
    # The task doesn't require reproducibility. Let's use random.
    rng = random.Random()
    
    # Total budget for restarts
    # We have used some evals in the first SIQI.
    # Let's try up to 10-20 restarts depending on remaining budget.
    
    max_restarts = 15
    
    for i in range(max_restarts):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
            break
            
        # Choose a scale
        scale = rng.choice(scales)
        # Choose a random offset in [-1, 1]
        offset = rng.uniform(-1.0, 1.0)
        x_start = x0 + offset * scale * base_scale
        
        # Ensure x_start is not too close to x0 (to avoid duplicate work)
        if abs(x_start - x0) < 1e-10 * base_scale:
            continue
            
        x_res, f_res = siqi(x_start)
        if is_good(x_res, f_res):
            return x_res

    # If all SIQI attempts failed, return the best point found during the process
    return best_x

# EVOLVE-BLOCK-END
