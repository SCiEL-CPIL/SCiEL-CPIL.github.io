import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses Brent's method (scipy.optimize.brentq) with adaptive bracketing,
    and a deadline guard to prevent timeout.
    """
    x0 = float(x0)
    tol = 1e-8
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s

    call_count = 0

    def safe_f(x):
        nonlocal call_count
        if call_count >= 450:
            raise ArithmeticError("Too many calls")
        call_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except:
            return float('inf')

    # Track best point found
    best_x = x0
    best_f = safe_f(x0)
    if abs(best_f) <= tol:
        return x0

    # Try to find a bracket using adaptive expansion
    def find_bracket(x0, f):
        """Find [a, b] such that f(a)*f(b) < 0."""
        a = x0
        fa = f(a)
        if abs(fa) <= tol:
            return a, a  # Already a root
        
        # Try small steps first, then expand
        h = 1e-6 * max(1.0, abs(x0))
        steps = [1, -1, 2, -2, 5, -5, 10, -10, 50, -50, 100, -100, 1000, -1000]
        
        for s in steps:
            b = x0 + s * h
            fb = f(b)
            if abs(fb) <= tol:
                return b, b
            if fa * fb < 0:
                return a, b if fa < 0 else b, a
            # If the new point is better, make it our new center
            if abs(fb) < abs(fa):
                a = b
                fa = fb
                h = abs(b - x0)
        
        # Geometric expansion
        width = max(1.0, abs(x0)) * 0.1
        center = x0
        fc = f(center)
        
        for _ in range(40):
            a = center - width
            b = center + width
            fa = f(a)
            fb = f(b)
            if abs(fa) <= tol:
                return a, a
            if abs(fb) <= tol:
                return b, b
            if fa * fb < 0:
                return a, b if fa < 0 else b, a
            width *= 2
            if not math.isfinite(width) or width > 1e10:
                break
            if time.time() - start_time > time_limit - 5.0:
                break
        
        return None

    bracket = find_bracket(x0, safe_f)
    
    if bracket is not None:
        a, b = bracket
        if abs(a - b) < 1e-15:
            return a
        
        # Use scipy's brentq for superlinear convergence
        try:
            from scipy.optimize import brentq
            if time.time() - start_time > time_limit - 2.0:
                # Not enough time, return best of endpoints
                fa, fb = safe_f(a), safe_f(b)
                return a if abs(fa) < abs(fb) else b
            
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            # Fall back to bisection
            pass
        
        # Bisection fallback
        fa = safe_f(a)
        fb = safe_f(b)
        for _ in range(100):
            if time.time() - start_time > time_limit - 1.0:
                break
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if abs(fm) <= tol:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-13:
                break
        
        return a if abs(fa) < abs(fb) else b
    else:
        # No bracket found, use secant/Newton-like approach
        x1 = x0
        f1 = safe_f(x1)
        if abs(f1) <= tol:
            return x1
        
        h = 1e-4 * max(1.0, abs(x0))
        x2 = x0 + h
        f2 = safe_f(x2)
        if abs(f2) <= tol:
            return x2
        
        # Secant iterations
        for _ in range(50):
            if time.time() - start_time > time_limit - 1.0:
                break
            denom = f2 - f1
            if abs(denom) < 1e-15:
                break
            x3 = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x3):
                break
            f3 = safe_f(x3)
            if abs(f3) <= tol:
                return x3
            if abs(f3) < abs(f2):
                x1, f1 = x2, f2
                x2, f3 = x3, f3  # Keep f3 as f2
            else:
                break
        
        # Return best point
        return x0 if abs(f1) < abs(f2) else x2
