"""Root finder using adaptive bracketing and inverse quadratic interpolation."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using adaptive bracketing and inverse quadratic interpolation.
    Returns x such that |f(x)| <= 1e-8, or the best approximation found within time limits.
    """
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s kill
    TARGET = 1e-8
    MAX_CALLS = 500

    x = float(x0)
    fx = f(x)
    if abs(fx) <= TARGET:
        return x

    # Initialize bracket search
    # We need to find a, b such that f(a)*f(b) < 0
    a = b = None
    fa = fb = None
    
    # Try small steps first, then expand
    step = 1e-6
    max_expansion = 1e6
    
    # Phase 1: Find a bracket
    calls = 1  # We already called f(x)
    
    # Try to find a bracket by expanding from x0
    found_bracket = False
    for i in range(100):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Try right side
        if b is None or fb * fa > 0:
            x_try = x + step
            f_try = f(x_try)
            calls += 1
            if abs(f_try) <= TARGET:
                return x_try
            if fa * f_try < 0:
                b, fb = x_try, f_try
                found_bracket = True
                break
            elif b is None:
                b, fb = x_try, f_try
            else:
                # Update b to the point with smaller |f| if same sign
                if abs(f_try) < abs(fb):
                    b, fb = x_try, f_try
        
        # Try left side
        if a is None or fa * fb > 0:
            x_try = x - step
            f_try = f(x_try)
            calls += 1
            if abs(f_try) <= TARGET:
                return x_try
            if fa * f_try < 0:
                a, fa = x_try, f_try
                found_bracket = True
                break
            elif a is None:
                a, fa = x_try, f_try
            else:
                # Update a to the point with smaller |f| if same sign
                if abs(f_try) < abs(fa):
                    a, fa = x_try, f_try
        
        step *= 2.0
        if step > max_expansion:
            break

    if not found_bracket:
        # If no bracket found, return the point with smallest |f|
        # This is a fallback; ideally we should have found one for continuous oscillatory functions
        # But for safety, return best effort
        candidates = [(x, fx)]
        if a is not None:
            candidates.append((a, fa))
        if b is not None:
            candidates.append((b, fb))
        best_x, best_f = min(candidates, key=lambda p: abs(p[1]))
        return best_x

    # Phase 2: Refine using IQI and Bisection
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    for _ in range(100):
        if time.time() - start_time > TIME_LIMIT:
            break
        if calls >= MAX_CALLS - 5:
            break

        # Check if we already have a root
        if abs(fa) <= TARGET:
            return a
        if abs(fb) <= TARGET:
            return b

        # Inverse Quadratic Interpolation
        # We need a third point. Use the midpoint or a previously evaluated point.
        # For simplicity, we'll use a secant-like approach or just bisection if IQI is unstable.
        # A common robust method: use the last two points and the midpoint of the bracket if needed.
        
        # Let's try a simpler robust approach: Bisection + Secant
        # Since we have a bracket, bisection is guaranteed to converge.
        # Secant can be faster. Let's alternate or use a hybrid.
        
        # Midpoint
        mid = (a + b) / 2.0
        f_mid = f(mid)
        calls += 1
        
        if abs(f_mid) <= TARGET:
            return mid

        if fa * f_mid <= 0:
            b, fb = mid, f_mid
        else:
            a, fa = mid, f_mid

        # Try Secant Step
        if fa != fb:
            x_sec = b - fb * (b - a) / (fb - fa)
            # Check if x_sec is within the bracket
            if a < x_sec < b:
                f_sec = f(x_sec)
                calls += 1
                if abs(f_sec) <= TARGET:
                    return x_sec
                if fa * f_sec <= 0:
                    b, fb = x_sec, f_sec
                else:
                    a, fa = x_sec, f_sec

        # If bracket is small enough, stop
        if abs(b - a) < 1e-12:
            break

    # Return the point with smallest |f| in the final bracket
    if abs(fa) < abs(fb):
        return a
    else:
        return b
# EVOLVE-BLOCK-END
