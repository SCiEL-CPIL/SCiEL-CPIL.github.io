"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a hybrid strategy:
    1. Check if x0 is already a root.
    2. Try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       If found, use scipy.optimize.brentq which is robust and efficient.
    3. If no bracket is found, use a safeguarded Newton's method with finite difference derivative.
    Includes a deadline guard to prevent timeout.
    """
    TOL = 1e-8
    DEADLINE = 145.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def safe_f(x):
        try:
            val = f(x)
            if not is_finite(val):
                return None
            return float(val)
        except:
            return None

    # Track best solution found
    best_x = None
    best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if is_finite(x) and is_finite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if best_x is not None and abs(best_fx) <= TOL:
            return True
        return False

    # Initial check
    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is None:
        return x0

    if abs(fx0) <= TOL:
        return x0
    
    update_best(x0, fx0)

    # Strategy 1: Bracketing + Brent's Method
    # Try to find a bracket by expanding from x0
    # Start with a small scale to avoid wasting calls on distant evaluations if root is nearby
    a = x0 - 0.1 * max(1.0, abs(x0))
    b = x0 + 0.1 * max(1.0, abs(x0))
    fa = None
    fb = None
    
    bracket_found = False
    # Expand bracket
    for i in range(40):
        if not check_deadline():
            break
            
        fa = safe_f(a)
        fb = safe_f(b)
        
        if fa is None or fb is None:
            break
            
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
            
        if fa * fb < 0:
            bracket_found = True
            break
            
        # Expand
        # Double the distance from x0
        a = x0 - (x0 - a) * 2
        b = x0 + (b - x0) * 2
        
        # Safety: don't expand to infinity
        if abs(a) > 1e10 or abs(b) > 1e10:
            break

    if bracket_found:
        try:
            # brentq is very robust and typically uses ~10-15 function calls
            root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if is_finite(root):
                return root
        except Exception:
            pass # Fall through to Newton

    # Strategy 2: Safeguarded Newton's Method with Secant fallback
    # This is more efficient than pure Newton (1 call/step vs 2) when derivative is not available
    # but we use Newton logic for stability if we have a good derivative estimate.
    # Actually, let's use a hybrid: Newton if we can estimate derivative well, else Secant.
    # For simplicity and call efficiency, let's use a damped Newton with finite diff.
    # But to save calls, we can try Secant first if we have two points.
    # We have (x0, fx0). Let's get another point.
    
    # Reuse fa, fb if they were close? No, let's just do a standard safeguarded Newton.
    # To minimize calls, we use the fact that we already have fx0.
    
    x = x0
    fx = fx0
    
    # We need a second point for Secant or a derivative for Newton.
    # Let's try Newton with finite difference first, as it converges quadratically.
    # But each step costs 2 calls.
    # Secant costs 1 call per step after the first.
    # Let's implement a robust Newton that falls back to bisection-like steps if needed.
    
    for i in range(60):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        
        if fx_plus is None:
            break
            
        if abs(fx_plus) <= TOL:
            return x + h
            
        dfx = (fx_plus - fx) / h
        
        if dfx == 0 or not is_finite(dfx):
            # Derivative is zero or bad, try a small step
            x_new = x + 1e-4 * max(1.0, abs(x))
            fx = safe_f(x_new)
            if fx is None:
                break
            x = x_new
            continue
            
        x_new = x - fx / dfx
        
        # Safeguard: if step is too large or non-finite, dampen
        if not is_finite(x_new) or abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            x_new = x - 0.5 * (fx / dfx)
            
        if not is_finite(x_new):
            break
            
        # Evaluate new point
        fx_new = safe_f(x_new)
        
        if fx_new is None:
            # Dampen further
            x_new = x - 0.1 * (fx / dfx)
            if not is_finite(x_new):
                break
            fx_new = safe_f(x_new)
            if fx_new is None:
                break
                
        if abs(fx_new) <= TOL:
            return x_new
            
        x = x_new
        fx = fx_new
        
    # Return best found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
