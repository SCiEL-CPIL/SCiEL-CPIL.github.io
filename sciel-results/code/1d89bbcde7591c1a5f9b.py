import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using adaptive bracketing and Illinois/Regula Falsi hybrid.
    Designed for oscillatory functions where Newton might diverge.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit, plenty for <500 calls
    
    x = float(x0)
    
    # 1. Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # 2. Adaptive Bracketing Phase
    # We need an interval [a, b] such that f(a) and f(b) have opposite signs.
    # Start by expanding from x0.
    a = x
    fa = fx
    
    # Try small steps first, then exponential expansion
    # Step sizes: 1e-2, 1e-1, 1, 10, 100...
    # Also try negative directions.
    
    # We'll maintain a list of candidate brackets or just search linearly/exponentially.
    # To minimize calls, we search in a specific pattern.
    
    # Let's try a few fixed offsets first to see if we can find a sign change quickly.
    # Common scales for "fixed equations" might be O(1).
    
    search_steps = [
        1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0, 
        100.0, -100.0, 1000.0, -1000.0
    ]
    
    b = None
    fb = None
    
    for step in search_steps:
        if time.time() > deadline - 1.0:
            break
        x_try = x + step
        f_try = f(x_try)
        if abs(f_try) <= 1e-8:
            return x_try
        
        # Check for sign change with current best 'a' (which is x0)
        # Or check if this new point creates a bracket with any previous point?
        # For simplicity and speed, let's just check against x0 (a) first.
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        
        # If we haven't found a bracket yet, maybe we should update 'a' 
        # to keep the interval small? No, for oscillatory, expanding is safer 
        # to catch a zero crossing if the function is bounded.
        # However, if the function is monotonic-ish locally, we might miss it 
        # if we jump over it. But "oscillatory" implies multiple roots.
        # A sign change is the safest bet for bisection.
        
    # If no bracket found with simple steps, try a denser search near x0?
    # Or expand further?
    if b is None:
        # Fallback: try expanding range more aggressively or using a different heuristic
        # Since we are limited in calls, let's try a few more large steps
        for mult in [1e4, -1e4, 1e5, -1e5]:
            if time.time() > deadline - 1.0:
                break
            x_try = x + mult
            f_try = f(x_try)
            if abs(f_try) <= 1e-8:
                return x_try
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
    
    # If we still don't have a bracket, we are in trouble.
    # We could return the best point found so far, but it likely fails |f(x)|<=1e-8.
    # Given the task guarantees roots exist for the 3 fixed equations, 
    # and x0 is a "starting point", it's likely close.
    # If bracketing fails completely, we might just have to guess or return x0.
    # But let's assume we found one. If not, return x0 to avoid exception, 
    # though it will likely fail the accuracy check.
    if b is None:
        return x

    # Ensure a and b are ordered (a < b) for standard bisection logic, 
    # though Illinois method works with signs.
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # 3. Illinois/Regula Falsi Hybrid Root Finding
    # Illinois method is a variant of Regula Falsi that avoids slow convergence 
    # on one side by shrinking the interval on the side that keeps having the 
    # same sign.
    
    # Initialize
    # We know fa and fb have opposite signs.
    # Let's keep track of the "best" candidate (lowest |f|) just in case.
    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))
    
    # Illinois Method:
    # 1. Calculate x_new using linear interpolation (Regula Falsi).
    # 2. If sign(f_new) == sign(fa), replace fa with fa/2 (or similar damping).
    # 3. If sign(f_new) == sign(fb), replace fb with fb/2.
    # This ensures the interval shrinks on both sides eventually.
    
    # Standard Illinois:
    # x_new = (a*fb - b*fa) / (fb - fa)
    # if f(x_new) * fa < 0:
    #     b = x_new; fb = f(x_new)
    #     fa = fa / 2  # Dampen the side that didn't change sign
    # else:
    #     a = x_new; fa = f(x_new)
    #     fb = fb / 2
    
    # Note: We must be careful with the damping factor. 0.5 is standard.
    
    for i in range(100): # Max 100 iterations is plenty for 1e-8 precision
        if time.time() > deadline - 0.5:
            break
        
        # Check if we've already converged
        if best_f <= 1e-8:
            return best_x
            
        # Calculate next point
        # Avoid division by zero if fa == fb (shouldn't happen if signs differ)
        if fa == fb:
            break
            
        denom = fb - fa
        if denom == 0:
            break
            
        x_new = (a * fb - b * fa) / denom
        f_new = f(x_new)
        
        # Update best candidate
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Determine which side to shrink/dampen
        if f_new * fa < 0:
            # Root is in [a, x_new]
            b = x_new
            fb = f_new
            # Dampen fa to ensure convergence from the other side
            fa = fa / 2.0
        else:
            # Root is in [x_new, b]
            a = x_new
            fa = f_new
            # Dampen fb
            fb = fb / 2.0
            
        # Safety check for interval size
        if abs(b - a) < 1e-15:
            break

    return best_x
# EVOLVE-BLOCK-END
