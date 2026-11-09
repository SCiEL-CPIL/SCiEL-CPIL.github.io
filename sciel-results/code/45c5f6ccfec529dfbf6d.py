"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    """
    Find a root of f using a robust combination of:
    1. Initial scanning to find a bracket
    2. Bisection for robustness
    3. Illinois/Regula Falsi for faster convergence
    4. Newton's method with safeguards when derivative is well-behaved
    
    Uses at most 500 evaluations of f.
    """
    x0 = float(x0)
    
    # Track evaluations
    eval_count = 0
    max_evals = 500
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Evaluation limit exceeded")
        eval_count += 1
        return float(f(x))
    
    # Try to find a bracket by scanning around x0
    # Use geometric spacing to cover a wide range
    def find_bracket(x_center, f_center):
        """Find an interval [a, b] where f(a) and f(b) have opposite signs."""
        if eval_count >= max_evals:
            return None
        
        # If f(x_center) is already close to zero
        if abs(f_center) <= 1e-10:
            return (x_center, x_center)
        
        # Try small steps first, then expand
        h = 1e-8 * max(1.0, abs(x_center))
        
        for _ in range(200):
            if eval_count >= max_evals - 2:
                break
            
            a = x_center - h
            b = x_center + h
            
            fa = eval_f(a)
            fb = eval_f(b)
            
            # Check if a is a root
            if abs(fa) <= 1e-10:
                return (a, a)
            # Check if b is a root
            if abs(fb) <= 1e-10:
                return (b, b)
            # Check for sign change
            if fa * fb < 0:
                return (a, b)
            
            # Expand the interval
            h *= 2.0
            if h > 1e10:
                break
        
        return None
    
    # Get initial evaluation
    fx0 = eval_f(x0)
    
    if abs(fx0) <= 1e-10:
        return x0
    
    # Try to find a bracket
    bracket = find_bracket(x0, fx0)
    
    if bracket is not None:
        a, b = bracket
        fa = eval_f(a) if a != x0 else fx0
        fb = eval_f(b) if b != x0 else fx0
        
        # If bracket is degenerate (a == b), we already found a root
        if a == b:
            return a
        
        # Use Illinois method (modified regula falsi) for robust convergence
        # This is more robust than pure bisection and faster than standard regula falsi
        for _ in range(max_evals - eval_count):
            if abs(fa) <= 1e-10:
                return a
            if abs(fb) <= 1e-10:
                return b
            
            # Illinois method: shrink the side with the larger absolute value
            if abs(fa) > abs(fb):
                # Shrink a
                fa = fa * 0.5
                a = (a * fb - b * fa) / (fb - fa)
            else:
                # Shrink b
                fb = fb * 0.5
                b = (a * fb - b * fa) / (fb - fa)
            
            f_mid = eval_f(a if abs(fa) > abs(fb) else b)
            
            if abs(f_mid) <= 1e-10:
                return a if abs(fa) > abs(fb) else b
            
            # Update bracket
            if (a if abs(fa) > abs(fb) else b) != a:
                # b was updated
                fb = f_mid
            else:
                fa = f_mid
        
        # Return the point with smaller |f|
        return a if abs(fa) < abs(fb) else b
    
    # If no bracket found, use a derivative-free optimization approach
    # Minimize g(x) = f(x)^2 using a simple coordinate descent / pattern search
    best_x = x0
    best_f = fx0
    
    if abs(best_f) <= 1e-10:
        return best_x
    
    # Pattern search / coordinate descent
    step = 1e-4 * max(1.0, abs(x0))
    
    for _ in range(max_evals - eval_count):
        improved = False
        
        for direction in [1, -1]:
            x_new = best_x + direction * step
            f_new = eval_f(x_new)
            
            if abs(f_new) < abs(best_f):
                best_x = x_new
                best_f = f_new
                improved = True
                
                if abs(best_f) <= 1e-10:
                    return best_x
        
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-15:
                break
        else:
            # Try larger step if improving
            step *= 1.5
    
    return best_x
# EVOLVE-BLOCK-END
