"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using a hybrid bisection/Newton (Brent-like) strategy
    with global bracketing. Designed for ill-conditioned equations.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Track best solution found
    best_x = float(x0)
    best_f = f(best_x)
    best_abs = abs(best_f)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax
    
    # Step 1: Try to find a bracket [a, b] where f(a)*f(b) < 0
    # Start near x0 and expand exponentially in both directions
    x = float(x0)
    fx = best_f
    
    # If already good enough
    if best_abs <= 1e-8:
        return best_x
    
    # Try to find a bracket
    a = b = None
    fa = fb = None
    
    # Try expanding from x0
    step = 1e-4
    for i in range(60):
        if time.time() > deadline:
            break
        
        # Try right side
        xr = x + step
        fr = f(xr)
        update_best(xr, fr)
        
        if fx * fr < 0:
            a, b = x, xr
            fa, fb = fx, fr
            break
        elif abs(fr) <= 1e-8:
            return xr
        
        # Try left side
        xl = x - step
        fl = f(xl)
        update_best(xl, fl)
        
        if fx * fl < 0:
            a, b = xl, x
            fa, fb = fl, fx
            break
        elif abs(fl) <= 1e-8:
            return xl
        
        step *= 2.0
    
    # If no bracket found, try a few more strategies
    if a is None:
        # Try a wider search with different starting points
        for offset in [0.1, 1.0, 10.0, 100.0]:
            if time.time() > deadline:
                break
            x_try = x0 + offset
            f_try = f(x_try)
            update_best(x_try, f_try)
            if abs(f_try) <= 1e-8:
                return x_try
            
            x_try2 = x0 - offset
            f_try2 = f(x_try2)
            update_best(x_try2, f_try2)
            if abs(f_try2) <= 1e-8:
                return x_try2
            
            if f_try * f_try2 < 0:
                a, b = x_try, x_try2
                fa, fb = f_try, f_try2
                break
    
    # If still no bracket, fall back to Newton with safeguards
    if a is None:
        x = float(x0)
        for i in range(100):
            if time.time() > deadline:
                break
            fx = f(x)
            update_best(x, fx)
            if abs(fx) <= 1e-8:
                return x
            
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0 or not math.isfinite(d):
                # Try a small perturbation
                x += 1e-6 * (1 if i % 2 == 0 else -1)
                continue
            
            x_new = x - fx / d
            # Safeguard: don't jump too far
            if not math.isfinite(x_new):
                break
            # Limit step size
            max_step = 10.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + math.copysign(max_step, x_new - x)
            
            x = x_new
        
        return best_x
    
    # Step 2: Use hybrid bisection/Newton within the bracket [a, b]
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # Verify bracket
    if fa * fb > 0:
        return best_x
    
    for i in range(200):
        if time.time() > deadline:
            break
        
        # Check convergence
        if best_abs <= 1e-8:
            return best_x
        
        # Bisection midpoint
        m = 0.5 * (a + b)
        fm = f(m)
        update_best(m, fm)
        
        if abs(fm) <= 1e-8:
            return m
        
        # Decide which sub-bracket to keep
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
        
        # Try Newton step for acceleration
        # Estimate derivative using secant or finite difference
        h = 1e-8 * max(1.0, abs(m))
        d = (f(m + h) - fm) / h
        
        if d != 0 and math.isfinite(d):
            x_newton = m - fm / d
            
            # Check if Newton step stays within bracket
            if a < x_newton < b:
                # Evaluate at Newton point
                f_newton = f(x_newton)
                update_best(x_newton, f_newton)
                
                if abs(f_newton) <= 1e-8:
                    return x_newton
                
                # Update bracket based on Newton point
                if fa * f_newton < 0:
                    b = x_newton
                    fb = f_newton
                else:
                    a = x_newton
                    fa = f_newton
        
        # Check if bracket is small enough
        if abs(b - a) < 1e-15 * max(1.0, abs(m)):
            break
    
    return best_x
# EVOLVE-BLOCK-END
