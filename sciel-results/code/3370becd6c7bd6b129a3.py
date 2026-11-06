"""Root finding via hybrid secant-bisection with Newton acceleration."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: secant method with bracket detection,
    bisection for guaranteed convergence, and Newton acceleration.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 480  # Safety margin for 500 limit
    
    eval_count = [0]
    
    def eval_f(x):
        """Evaluate f with counting and deadline check."""
        if eval_count[0] >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count[0] += 1
        try:
            return f(x)
        except Exception:
            return None
    
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Try to establish a bracket around x0
    # Start with a small interval and expand if needed
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    f1 = eval_f(x1)
    
    if f1 is None:
        return x
    if abs(f1) <= 1e-8:
        return x1
    
    # Check if we have a bracket [x, x1]
    bracket = None
    if fx * f1 < 0:
        bracket = (x, x1, fx, f1)
    else:
        # Expand the interval to find a bracket
        # Try expanding in both directions
        a, b = None, None
        fa, fb = None, None
        
        # Expand to the right
        step = h
        for _ in range(20):
            if eval_count[0] >= max_evals or time.time() > deadline:
                break
            x_new = x + step
            f_new = eval_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new
            if fx * f_new < 0:
                a, b = x, x_new
                fa, fb = fx, f_new
                break
            step *= 2
            if step > 1e6 * max(1.0, abs(x)):
                break
        
        # If no bracket found expanding right, try left
        if a is None:
            step = h
            for _ in range(20):
                if eval_count[0] >= max_evals or time.time() > deadline:
                    break
                x_new = x - step
                f_new = eval_f(x_new)
                if f_new is None:
                    break
                if abs(f_new) <= 1e-8:
                    return x_new
                if fx * f_new < 0:
                    a, b = x_new, x
                    fa, fb = f_new, fx
                    break
                step *= 2
                if step > 1e6 * max(1.0, abs(x)):
                    break
        
        if a is not None:
            bracket = (a, b, fa, fb)
    
    # Track best solution found so far
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    if bracket is not None:
        a, b, fa, fb = bracket
        # Ensure fa corresponds to a and fb to b
        if fa * fb > 0:
            # Shouldn't happen, but swap if needed
            a, b = b, a
            fa, fb = fb, fa
        
        # Use hybrid secant-bisection
        # Initialize secant points
        x0_sec = a
        x1_sec = b
        f0_sec = fa
        f1_sec = fb
        
        for _ in range(200):
            if eval_count[0] >= max_evals or time.time() > deadline:
                break
            
            # Check convergence
            if abs(fa) <= 1e-8:
                best_x = a
                best_fx = fa
                best_abs = abs(fa)
                return a
            if abs(fb) <= 1e-8:
                best_x = b
                best_fx = fb
                best_abs = abs(fb)
                return b
            
            # Try Newton step if we can estimate derivative
            # Use secant derivative
            if x1_sec != x0_sec:
                d = (f1_sec - f0_sec) / (x1_sec - x0_sec)
                if abs(d) > 1e-15:
                    # Newton step from the point with smaller |f|
                    if abs(fa) <= abs(fb):
                        x_newton = a - fa / d
                    else:
                        x_newton = b - fb / d
                    
                    # Check if Newton step is within bracket
                    if min(a, b) < x_newton < max(a, b):
                        f_newton = eval_f(x_newton)
                        if f_newton is None:
                            break
                        if abs(f_newton) <= 1e-8:
                            best_x = x_newton
                            best_fx = f_newton
                            best_abs = abs(f_newton)
                            return x_newton
                        
                        # Update bracket
                        if fa * f_newton < 0:
                            b, fb = x_newton, f_newton
                        else:
                            a, fa = x_newton, f_newton
                        
                        # Update secant points
                        x0_sec, x1_sec = a, b
                        f0_sec, f1_sec = fa, fb
                        continue
                    else:
                        # Newton step outside bracket, use bisection
                        pass
            
            # Bisection step
            mid = 0.5 * (a + b)
            f_mid = eval_f(mid)
            if f_mid is None:
                break
            if abs(f_mid) <= 1e-8:
                best_x = mid
                best_fx = f_mid
                best_abs = abs(f_mid)
                return mid
            
            if fa * f_mid < 0:
                b, fb = mid, f_mid
            else:
                a, fa = mid, f_mid
            
            # Update secant points
            x0_sec, x1_sec = a, b
            f0_sec, f1_sec = fa, fb
    
    # If no bracket found or bracket method didn't converge, try secant method
    # without bracketing
    if eval_count[0] < max_evals and time.time() < deadline:
        # Use secant method
        x_prev = x
        f_prev = fx
        x_curr = x1
        f_curr = f1
        
        for _ in range(200):
            if eval_count[0] >= max_evals or time.time() > deadline:
                break
            
            if abs(f_curr) < best_abs:
                best_x = x_curr
                best_fx = f_curr
                best_abs = abs(f_curr)
            if abs(f_prev) < best_abs:
                best_x = x_prev
                best_fx = f_prev
                best_abs = abs(f_prev)
            
            if best_abs <= 1e-8:
                return best_x
            
            if x_curr == x_prev:
                break
            
            d = (f_curr - f_prev) / (x_curr - x_prev)
            if abs(d) < 1e-15:
                break
            
            x_next = x_curr - f_curr / d
            
            # Damping if step is too large
            step = abs(x_next - x_curr)
            max_step = 10.0 * max(1.0, abs(x_curr))
            if step > max_step:
                x_next = x_curr + max_step * (1.0 if step > 0 else -1.0)
            
            f_next = eval_f(x_next)
            if f_next is None:
                break
            
            if abs(f_next) <= 1e-8:
                best_x = x_next
                best_fx = f_next
                best_abs = abs(f_next)
                return x_next
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next
    
    return best_x
# EVOLVE-BLOCK-END
