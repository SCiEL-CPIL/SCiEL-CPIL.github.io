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

    # Initial step for Secant/Illinois - smaller step for better local bracket detection
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
    
    # If no bracket, try to find one by expanding in both directions
    if not has_bracket:
        # Try expanding in both directions, prefer the side with smaller |f|
        best_left = a
        best_left_f = fa
        best_right = b
        best_right_f = fb
        step = h
        
        for _ in range(60):
            if time.time() > deadline:
                break
            
            # Expand left
            x_try = best_left - step
            f_try = f(x_try)
            if abs(f_try) <= 1e-8:
                return x_try
            if fa * f_try < 0:
                b = a
                fb = fa
                a = x_try
                fa = f_try
                has_bracket = True
                break
            if abs(f_try) < abs(fa):
                best_left = x_try
                best_left_f = f_try
                fa = f_try
                a = x_try
            
            # Expand right
            x_try_r = best_right + step
            f_try_r = f(x_try_r)
            if abs(f_try_r) <= 1e-8:
                return x_try_r
            if fb * f_try_r < 0:
                a = b
                fa = fb
                b = x_try_r
                fb = f_try_r
                has_bracket = True
                break
            if abs(f_try_r) < abs(fb):
                best_right = x_try_r
                best_right_f = f_try_r
                fb = f_try_r
                b = x_try_r
            
            step *= 2.0

    # Track the best x found so far
    best_x = x if abs(fx) < abs(f_prev) else x_prev
    best_f = min(abs(fx), abs(f_prev))

    for i in range(200): # Max iterations safeguard
        if time.time() > deadline:
            return best_x

        # Choose method
        if has_bracket:
            # Illinois Algorithm
            # Secant step
            if fb - fa == 0:
                # Flat line? Use bisection
                x_next = (a + b) / 2.0
            else:
                x_next = b - fb * (b - a) / (fb - fa)
            
            # Safeguard: if x_next is outside [a,b], use bisection
            if x_next <= a or x_next >= b:
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
                x_next = b + 1e-6 * max(1.0, abs(b))
            else:
                x_next = b - fb * (b - a) / (fb - fa)
            
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

    # Polish pass: use secant with tighter tolerance
    if best_f > 1e-10:
        for i in range(20):
            if time.time() > deadline:
                break
            if fb - fa == 0:
                break
            x_next = b - fb * (b - a) / (fb - fa)
            f_next = f(x_next)
            if abs(f_next) <= 1e-8:
                return x_next
            if abs(f_next) < best_f:
                best_f = abs(f_next)
                best_x = x_next
            a = b
            fa = fb
            b = x_next
            fb = f_next

    return best_x
