import math
import time

def find_root(f, x0):
    # Deadline guard: 150s wall clock limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Initial step for Secant/Illinois
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x - h
    f_prev = f(x_prev)
    
    if abs(f_prev) <= 1e-8:
        return x_prev

    # Check for bracket
    a = x_prev
    b = x
    fa = f_prev
    fb = fx
    has_bracket = (fa * fb < 0)
    
    # If no bracket, try to find one by expanding
    if not has_bracket:
        # Simple expansion strategy to try to find a sign change
        # This is risky for call count, but necessary for robustness if x0 is far.
        # However, the task implies x0 is a "starting point" for a root finder, 
        # suggesting local convergence is expected.
        # We will proceed with Secant but monitor for divergence.
        pass

    # Illinois/Secant Hybrid
    # We track the best x found so far to return if we run out of time or fail
    best_x = x if abs(fx) < abs(f_prev) else x_prev
    best_f = min(abs(fx), abs(f_prev))

    for i in range(100): # Max iterations safeguard
        if time.time() > deadline:
            return best_x

        # Choose method
        if has_bracket:
            # Illinois Algorithm
            # Secant step
            if fb - fa == 0:
                # Flat line? Break
                break
            x_next = b - fb * (b - a) / (fb - fa)
            
            # Safeguard: if x_next is outside [a,b], use bisection
            if x_next < a or x_next > b:
                x_next = (a + b) / 2.0
            else:
                # Illinois modification: if x_next is closer to a, scale fa
                # Standard Illinois: if |x_next - a| < |x_next - b|, fa = fa * 0.5
                if abs(x_next - a) < abs(x_next - b):
                    fa *= 0.5
                else:
                    fb *= 0.5
                    
            f_next = f(x_next)
            
            if abs(f_next) <= 1e-8:
                return x_next
            
            if abs(f_next) < best_f:
                best_f = abs(f_next)
                best_x = x_next

            # Update bracket
            if fa * f_next <= 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
        else:
            # Secant Method
            if fb - fa == 0:
                # Derivative is zero, try a small perturbation
                x_next = b + 1e-4
            else:
                x_next = b - fb * (b - a) / (fb - fa)
            
            # Divergence check: if the new point is "worse" than the best 
            # and we are not making progress, we might be diverging.
            # For now, just proceed.
            
            f_next = f(x_next)
            
            if abs(f_next) <= 1e-8:
                return x_next
            
            if abs(f_next) < best_f:
                best_f = abs(f_next)
                best_x = x_next

            # Update points for next secant step
            a = b
            fa = fb
            b = x_next
            fb = f_next
            
            # Check if we can establish a bracket now
            if fa * fb < 0:
                has_bracket = True
                # Ensure a, b are ordered for bracket logic
                # Illinois logic assumes a, b are endpoints
                if a > b:
                    a, b = b, a
                    fa, fb = fb, fa

    return best_x
