import math
import time
import scipy.optimize

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses adaptive bracketing followed by SciPy's Brent's method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_up():
        return time.time() > deadline

    x0 = float(x0)
    
    # 1. Initial Evaluation
    try:
        fx0 = f(x0)
    except Exception:
        return x0

    if not math.isfinite(fx0):
        return x0

    if abs(fx0) <= 1e-8:
        return x0

    # 2. Bracketing Search
    # We need to find [a, b] such that f(a) * f(b) < 0.
    # Start with a small step and expand.
    # For oscillatory functions, roots can be close. Start small.
    h = 1e-4
    a, b = None, None
    fa, fb = None, None
    
    # We already have (x0, fx0).
    # We will search left and right.
    # To save calls, we can alternate or just expand.
    # Let's expand a window [x0 - h, x0 + h].
    
    best_x = x0
    best_f = abs(fx0)
    
    max_expansions = 60 # 1e-4 * 2^60 is huge
    for i in range(max_expansions):
        if time_up():
            break
            
        x_left = x0 - h
        x_right = x0 + h
        
        # Evaluate left
        try:
            f_left = f(x_left)
            if math.isfinite(f_left):
                if abs(f_left) < best_f:
                    best_f = abs(f_left)
                    best_x = x_left
                if abs(f_left) <= 1e-8:
                    return x_left
                if fx0 * f_left < 0:
                    a, b = x_left, x0
                    fa, fb = f_left, fx0
                    break
        except Exception:
            pass

        # Evaluate right
        try:
            f_right = f(x_right)
            if math.isfinite(f_right):
                if abs(f_right) < best_f:
                    best_f = abs(f_right)
                    best_x = x_right
                if abs(f_right) <= 1e-8:
                    return x_right
                if fx0 * f_right < 0:
                    a, b = x0, x_right
                    fa, fb = fx0, f_right
                    break
        except Exception:
            pass
        
        h *= 2.0
        if h > 1e12:
            break

    if a is None or b is None:
        # No bracket found. Return best point found.
        return best_x

    # 3. Brent's Method
    # Use scipy.optimize.brentq
    # We must ensure we don't exceed call count.
    # brentq does not expose a call counter easily, but maxiter limits iterations.
    # Each iteration is 1 or 2 calls. maxiter=50 means <100 calls.
    # Total calls so far: ~2*i (bracketing) + 1 (initial).
    # If i is small (e.g., 5), we have ~11 calls. 50 more is fine.
    
    try:
        # xtol and rtol should be tight enough to guarantee |f(x)| <= 1e-8?
        # Not necessarily. |f(x)| depends on the derivative.
        # However, for oscillatory functions with moderate derivatives, tight x tolerance works.
        # We will verify the result.
        root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
        
        if math.isfinite(root):
            f_root = f(root)
            if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                return root
            # If brentq converged but |f(root)| > 1e-8, it might be a very flat root
            # or the tolerance was too loose relative to the function scale.
            # In that case, return the best of root and best_x
            if abs(f_root) < best_f:
                return root
            return best_x
    except Exception:
        # If brentq fails, return best point found during bracketing
        return best_x
