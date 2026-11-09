"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    """
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    max_evals = 500
    eval_count = 0
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')

    def safe_f(x):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if time.time() > deadline:
            raise TimeoutError()
        if eval_count >= max_evals:
            raise ValueError("max evals")
        eval_count += 1
        val = f(x)
        val = float(val)
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_fx:
                best_abs_fx = abs_val
                best_x = x
                best_fx = val
        return val

    # Try multiple strategies
    # Strategy 1: Secant method (no derivative needed, robust)
    # Strategy 2: Newton with finite difference
    # Strategy 3: Bisection if we can bracket

    x = float(x0)
    
    # First, evaluate at x0
    try:
        fx0 = safe_f(x)
    except (TimeoutError, ValueError):
        return best_x

    if abs(fx0) <= 1e-8:
        return x

    # Try to find a bracket using secant-like exploration
    # Then use a hybrid method: secant + bisection safeguard
    
    # Initialize secant with two points
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    x0_prev = x
    x1_prev = x1
    
    try:
        fx1 = safe_f(x1)
    except (TimeoutError, ValueError):
        return best_x

    if abs(fx1) <= 1e-8:
        return x1

    # Use Illinois / Pegasus method (robust secant variant)
    # Keep track of bracket if possible
    a, b = x0_prev, x1_prev
    fa, fb = fx0, fx1

    # Check if we have a bracket
    if fa * fb < 0:
        # We have a bracket, use bisection/secant hybrid
        pass
    else:
        # No bracket, try to find one by expanding
        step = h
        for _ in range(20):
            if time.time() > deadline or eval_count >= max_evals:
                break
            x_try = x0 + step
            try:
                fx_try = safe_f(x_try)
            except (TimeoutError, ValueError):
                break
            if abs(fx_try) <= 1e-8:
                return x_try
            if fa * fx_try < 0:
                a, b = x0, x_try
                fa, fb = fx0, fx_try
                break
            step *= 2
            if step > 1e6 * max(1.0, abs(x0)):
                break

    # Now use a robust hybrid: if bracketed, use bisection + secant; else use secant
    # with damping
    
    x_a, x_b = a, b
    fx_a, fx_b = fa, fb
    
    # If no bracket found, just use damped secant/Newton
    has_bracket = (fx_a * fx_b < 0)
    
    for iteration in range(450):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
            
        if has_bracket:
            # Bisection + secant hybrid (Illinois method)
            # Secant step
            if fx_a == fx_b:
                x_mid = 0.5 * (x_a + x_b)
            else:
                x_sec = x_b - fx_b * (x_b - x_a) / (fx_b - fx_a)
            
            # Check if secant is within bracket
            if x_a < x_sec < x_b:
                x_new = x_sec
            else:
                # Use bisection
                x_new = 0.5 * (x_a + x_b)
            
            try:
                fx_new = safe_f(x_new)
            except (TimeoutError, ValueError):
                break
                
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fx_a * fx_new < 0:
                x_b = x_new
                fx_b = fx_new
            else:
                x_a = x_new
                fx_a = fx_new
            
            # Illinois: if same side, halve the function value on the other side
            # This is implicitly handled by the bracket update
            
        else:
            # No bracket: use damped secant
            if abs(fx_b) < abs(fx_a):
                # x_b is closer
                pass
            
            # Secant step from (x_a, fx_a) and (x_b, fx_b)
            if fx_a == fx_b:
                x_new = 0.5 * (x_a + x_b)
            else:
                x_new = x_b - fx_b * (x_b - x_a) / (fx_b - fx_a)
            
            # Damping: if step is too large, damp it
            if abs(x_new - x_b) > 10 * max(1.0, abs(x_b)):
                x_new = x_b + 0.5 * (x_a - x_b)
            
            try:
                fx_new = safe_f(x_new)
            except (TimeoutError, ValueError):
                break
                
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Update: keep the point with smaller |f|
            if abs(fx_new) < abs(fx_a):
                x_a, fx_a = x_b, fx_b
                x_b, fx_b = x_new, fx_new
            else:
                x_a, fx_a = x_new, fx_new
            
            # Check if we now have a bracket
            if fx_a * fx_b < 0:
                has_bracket = True

        # Check convergence
        if abs(x_b - x_a) < 1e-14 * max(1.0, abs(x_b)):
            break

    return best_x
# EVOLVE-BLOCK-END
