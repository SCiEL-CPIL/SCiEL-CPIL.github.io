"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a robust hybrid bisection/secant approach.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals and time limit.
    """
    start_time = time.time()
    DEADLINE = 100.0  # Safety margin before 120s/150s kill
    MAX_EVALS = 500
    TOL = 1e-8
    
    eval_count = 0
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    # Track best point found
    best_x = float(x0)
    best_fx = None
    
    try:
        # Initial evaluation
        fx0 = call_f(float(x0))
        best_x = float(x0)
        best_fx = fx0
        
        if abs(best_fx) <= TOL:
            return best_x

        # Try to find a bracket [a, b] such that f(a)*f(b) < 0
        # Start with x0 as one endpoint
        a = float(x0)
        fa = fx0
        
        # Expand interval to find a sign change
        # We need to be careful not to waste too many evals here
        # Strategy: step out in both directions
        step = 1.0
        b = None
        fb = None
        
        # Try to the right
        for i in range(20):
            if not check_time():
                break
            x_test = a + step
            if eval_count >= MAX_EVALS:
                break
            fx_test = call_f(x_test)
            if fa * fx_test < 0:
                b = x_test
                fb = fx_test
                break
            # If we find a smaller |f|, update best
            if abs(fx_test) < abs(best_fx):
                best_x = x_test
                best_fx = fx_test
            step *= 2.0
            if step > 1e10:
                break
        
        if b is None:
            # Try to the left
            step = 1.0
            for i in range(20):
                if not check_time():
                    break
                x_test = a - step
                if eval_count >= MAX_EVALS:
                    break
                fx_test = call_f(x_test)
                if fa * fx_test < 0:
                    b = x_test
                    fb = fx_test
                    break
                if abs(fx_test) < abs(best_fx):
                    best_x = x_test
                    best_fx = fx_test
                step *= 2.0
                if step > 1e10:
                    break
        
        if b is None:
            # Could not find a bracket. 
            # This might happen if f(x0) is already very close or if the function is monotonic and doesn't cross 0 in the searched range.
            # In ill-conditioned problems, sometimes the "root" is a local minimum where f is small but not zero? 
            # The task says |f(x)| <= 1e-8. If we can't bracket, we return the best found.
            # However, usually roots are bracketable.
            # One more check: if |fx0| is small enough, we return.
            return best_x

        # Now we have [a, b] with fa * fb < 0
        # Ensure a has smaller |f| or just keep them
        # Hybrid Bisection/Secant Loop
        for i in range(MAX_EVALS - eval_count):
            if not check_time():
                break
            if eval_count >= MAX_EVALS:
                break
            
            # Check convergence
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
            # Calculate secant point
            denom = fb - fa
            if denom == 0:
                # Fall back to bisection
                c = (a + b) / 2.0
            else:
                # Secant step from a
                c = a - fa * (b - a) / denom
                # If secant step is outside or too close to a, use bisection or Illinois
                # To be robust, if c is not in (a, b), use bisection
                if c <= a or c >= b:
                    c = (a + b) / 2.0
                else:
                    # Check if secant step is "good" (reduces interval significantly)
                    # If not, maybe use Illinois point
                    # For simplicity and robustness, if secant is inside, use it.
                    # This is essentially Newton-Secant with Bisection fallback.
                    pass
            
            fc = call_f(c)
            
            # Update best
            if abs(fc) < abs(best_fx):
                best_x = c
                best_fx = fc
            
            if abs(fc) <= TOL:
                return c
            
            # Update bracket
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                
            # Convergence check on interval size
            if abs(b - a) < 1e-12:
                break
                
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
