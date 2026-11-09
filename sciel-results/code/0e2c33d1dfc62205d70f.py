import math

def find_root(f, x0):
    # Tolerance for success
    tol = 1e-8
    
    x = float(x0)
    fx = f(x)
    calls = 1
    
    if abs(fx) <= tol:
        return x

    # Try to find a bracket [a, b] such that f(a)*f(b) < 0
    # Start with small steps to catch nearby roots, then expand
    a = x
    fa = fx
    b = None
    fb = None
    
    # Small steps first
    for delta in [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]:
        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        calls += 1
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            break
        
        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        calls += 1
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            break
    
    if b is None:
        # If no bracket found, we are in trouble. 
        # For oscillatory functions, this is unlikely if we expanded enough.
        # Return the point with the smallest magnitude so far? 
        # Or raise an exception? The task says exception is a failure.
        # Let's try a few more random steps or just return x if nothing else works.
        # But better to try a wider search.
        # Actually, if we didn't find a sign change, maybe the function doesn't cross zero?
        # But the task implies roots exist.
        # Let's assume we found a bracket. If not, we fail this equation.
        # We'll return x to avoid exception, but it will likely fail the |f(x)| check.
        return x

    # Now we have a bracket [a, b] with fa*fb < 0
    # Use Illinois Algorithm (modified Bisection) for robustness and speed
    # Illinois is faster than Bisection and more robust than Secant.
    
    # Initialize
    x_a = a
    x_b = b
    f_a = fa
    f_b = fb
    
    # Limit iterations to ensure we stay under call budget
    # We have used ~10-20 calls for bracketing.
    # Budget remaining: ~480.
    # Illinois converges roughly linearly with factor ~0.6-0.8.
    # To get 1e-8, we need log(1e-8)/log(0.6) ~ 40 iterations if starting from width 1.
    # If width is larger, more iterations needed.
    
    for _ in range(400):
        # Check convergence
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b
            
        # Illinois Step:
        # If |f_a| < |f_b|, replace a with a weighted average towards b?
        # Standard Illinois:
        # If |f_a| < |f_b|:
        #    a = a - (f_a / (f_b - f_a)) * (b - a)  <-- This is Secant
        #    But Illinois replaces the point with SMALLER |f| with the Secant estimate?
        #    No, Illinois replaces the point with LARGER |f|?
        #    Let's check standard Illinois:
        #    If |f(a)| < |f(b)|, then b is the one to move?
        #    Actually, the algorithm keeps the bracket.
        #    Step:
        #    If |f_a| < |f_b|:
        #        a = a - (f_a / (f_b - f_a)) * (b - a)  # This is the secant point
        #        f_a = f(a)
        #        # Ensure a is still in bracket? 
        #        # Illinois actually replaces the endpoint with the secant point 
        #        # IF it is within the bracket. If not, it uses bisection?
        #        # Actually, classic Illinois just uses the secant formula 
        #        # to update the endpoint with the smaller function value magnitude.
        
        # Let's use a simpler Robust Secant:
        # 1. Calculate Secant x_s.
        # 2. If x_s is in [x_a, x_b], use it.
        # 3. Else, use Midpoint.
        
        if f_a == f_b:
            x_next = (x_a + x_b) / 2.0
        else:
            x_next = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            
        # Check if x_next is in bracket
        if (x_next < min(x_a, x_b)) or (x_next > max(x_a, x_b)):
            x_next = (x_a + x_b) / 2.0
            
        f_next = f(x_next)
        calls += 1
        
        if abs(f_next) <= tol:
            return x_next
            
        # Update Bracket
        if f_a * f_next < 0:
            x_b = x_next
            f_b = f_next
        else:
            x_a = x_next
            f_a = f_next
            
        # Convergence check on interval width
        if abs(x_b - x_a) < 1e-12:
            # Return the one with smaller |f|
            return x_a if abs(f_a) < abs(f_b) else x_b

    # If loop finishes, return best guess
    return x_a if abs(f_a) < abs(f_b) else x_b
