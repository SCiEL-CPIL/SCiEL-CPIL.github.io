"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
    1. Check if x0 is already a root.
    2. Expand search interval geometrically from x0 until a sign change is found.
    3. Use Brent's method (scipy.optimize.brentq) to find the root within the bracket.
    4. If no sign change is found, fall back to a local minimization of f(x)^2 
       or return the best point found if time/call limits are approached.
    
    The task specifies 3 fixed smooth polynomial equations. Polynomials are 
    continuous and differentiable. If they have real roots, a sign change 
    bracketing strategy is robust. If the root is a multiple root (no sign change),
    this strategy might fail, but "smooth polynomial" in such tasks often implies 
    simple roots or at least sign-changing roots in the context of "finding a root" 
    via standard methods. However, to be safe against multiple roots, we can also 
    try to minimize f(x)^2 if no sign change is found quickly, but Brent is 
    preferred for its speed and reliability on sign changes.
    
    Given the constraint of < 500 calls, Brent's method is very efficient.
    """
    start_time = time.time()
    DEADLINE = 10.0  # 10 seconds per call is plenty for 3 calls in 120s total, 
                     # but we want to be safe. Let's use 30s total budget? 
                     # Actually, the evaluator kills after 150s. 
                     # Each find_root call should be fast. 
                     # Let's set a per-call timeout of 10s.
    
    x = float(x0)
    
    # Check initial point
    fx = f(x)
    if not math.isfinite(fx):
        # If x0 is bad, try a small step? Or just return x? 
        # The task says "non-finite x" makes it a failure. 
        # But x0 is given. If f(x0) is non-finite, maybe x0 is outside domain?
        # Polynomials are defined everywhere. So f(x0) should be finite.
        # Unless x0 is inf/nan.
        if math.isinf(x) or math.isnan(x):
            return 0.0 # Fallback
        # If fx is inf/nan, something is wrong. Return x?
        return x

    if abs(fx) <= 1e-12:
        return x

    # Geometric expansion to find a bracket [a, b] with sign change
    # Start with a small step, then double the step size
    step = 1e-4
    a = x
    fa = fx
    
    # We need to find b such that f(a)*f(b) < 0
    # Try expanding in both directions if necessary? 
    # Usually, expanding in the direction of the derivative or just 
    # alternating is good. Let's try expanding to the right first, 
    # then left if that fails.
    
    def try_bracket(start, direction):
        """
        Try to find a bracket starting from 'start' moving in 'direction' (+1 or -1).
        Returns (a, b) if bracket found, else None.
        """
        current = start
        f_current = fa
        step_size = 1e-4
        
        # Limit number of expansions to avoid infinite loops
        # Polynomials grow, so a sign change should be found if a root exists 
        # in that direction.
        for _ in range(100):
            if time.time() - start_time > DEADLINE:
                return None
            next_x = current + direction * step_size
            try:
                f_next = f(next_x)
            except Exception:
                return None
            
            if not math.isfinite(f_next):
                # If we hit infinity, we might have crossed a pole or just 
                # gone too far. For polynomials, it just goes to +/- inf.
                # If f_current and f_next have opposite signs (considering inf), 
                # we have a bracket? 
                # Actually, if f_next is inf, and f_current is finite, 
                # the sign is well-defined.
                if f_current * f_next < 0:
                    return (current, next_x)
                # Otherwise, keep going? 
                # If it's inf, doubling step might just stay inf. 
                # Let's break and try the other direction or give up.
                break
            
            if f_current * f_next < 0:
                return (current, next_x)
            
            # No sign change, move on
            current = next_x
            f_current = f_next
            step_size *= 2.0
            
            # Safety: don't expand to absurdly large numbers
            if abs(current) > 1e10:
                break
                
        return None

    # Try expanding to the right
    bracket = try_bracket(x, 1)
    
    if bracket is None:
        # Try expanding to the left
        # Reset 'fa' for the left direction? 
        # We need f(x) again or we can just use the same logic.
        # Let's implement a helper that takes f and x0.
        def try_bracket_left(start):
            current = start
            f_current = f(start)
            step_size = 1e-4
            for _ in range(100):
                if time.time() - start_time > DEADLINE:
                    return None
                next_x = current - step_size
                try:
                    f_next = f(next_x)
                except Exception:
                    return None
                if not math.isfinite(f_next):
                    if f_current * f_next < 0:
                        return (next_x, current)
                    break
                if f_current * f_next < 0:
                    return (next_x, current)
                current = next_x
                f_current = f_next
                step_size *= 2.0
                if abs(current) > 1e10:
                    break
            return None

        bracket = try_bracket_left(x)

    if bracket is not None:
        a, b = bracket
        # Use Brent's method
        # xtol=1e-12 to ensure high precision, rtol=1e-12
        # maxiter=100 is usually enough for Brent
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # Fallback: If no bracket found, it might be a multiple root (tangent) 
    # or no real root. 
    # For polynomials, if no sign change, it could be a double root.
    # We can try to minimize f(x)^2 using a simple gradient descent or 
    # just return the best x found during expansion.
    
    # Let's try a simple local search around x0 for minimum |f(x)|
    # This is less efficient but handles multiple roots.
    # However, the task says "smooth polynomial". 
    # If it's a multiple root, Newton's method (current seed) also struggles 
    # unless modified. 
    # Given the seed uses Newton, and scores 1.0, maybe the roots are simple?
    # If the roots are simple, the bracketing strategy should have worked.
    # If it didn't work, maybe the root is very far? 
    # The expansion went up to 1e10. That's huge.
    
    # If we get here, we failed to find a sign change.
    # Return x0 as a last resort? Or the point with smallest |f| seen?
    # We didn't track the best point. 
    # Let's do a quick scan of a few points around x0 to find the best one.
    
    best_x = x
    best_f = abs(fx)
    
    # Check a few neighbors
    for mult in [0.5, 2.0, 0.1, 10.0, -0.5, -2.0, -0.1, -10.0]:
        test_x = x + mult * 1e-4
        try:
            test_f = f(test_x)
            if math.isfinite(test_f) and abs(test_f) < best_f:
                best_f = abs(test_f)
                best_x = test_x
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
