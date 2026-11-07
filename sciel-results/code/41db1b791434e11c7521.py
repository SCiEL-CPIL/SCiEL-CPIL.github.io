"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    x = float(x0)

    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # 2. Inverse Quadratic Interpolation (IQI) with Newton fallback
    # We maintain a history of points (x, fx).
    # If we have 3 points, we fit a quadratic to (fx, x) and find its root.
    # This is derivative-free and often faster than Newton for smooth functions.
    
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    # History of points for interpolation
    # We store (x, fx)
    history = [(x, fx)]
    
    # Initial step to get a second point if needed for secant/IQI
    # We'll try to generate a second point close to x0
    if len(history) < 2:
        # Try a small step
        h_init = 1e-4 * max(1.0, abs(x))
        x2 = x + h_init
        try:
            fx2 = f(x2)
            if math.isfinite(fx2):
                history.append((x2, fx2))
                if abs(fx2) < best_abs_fx:
                    best_x, best_fx, best_abs_fx = x2, fx2, abs(fx2)
                if abs(fx2) <= 1e-8:
                    return x2
        except Exception:
            pass
        
        if len(history) < 2:
            # Try opposite direction
            x2 = x - h_init
            try:
                fx2 = f(x2)
                if math.isfinite(fx2):
                    history.append((x2, fx2))
                    if abs(fx2) < best_abs_fx:
                        best_x, best_fx, best_abs_fx = x2, fx2, abs(fx2)
                    if abs(fx2) <= 1e-8:
                        return x2
            except Exception:
                pass

    # Main loop
    # We will iterate up to a limit, or until deadline.
    # Each iteration:
    # 1. Try IQI if we have 3 points.
    # 2. Else try Secant if we have 2 points.
    # 3. Else use a finite-difference Newton step (costs 1 extra call for derivative).
    # 4. Safeguard: if the new point is too far or |f| increases drastically, shrink or use bisection-like step.
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        current_x = history[-1][0]
        current_fx = history[-1][1]
        
        if abs(current_fx) <= 1e-8:
            return current_x
        
        # Update best
        if abs(current_fx) < best_abs_fx:
            best_x, best_fx, best_abs_fx = current_x, current_fx, abs(current_fx)

        x_new = None
        
        # Strategy 1: Inverse Quadratic Interpolation (IQI)
        if len(history) >= 3:
            # Use last 3 points
            x0_h, f0_h = history[-3]
            x1_h, f1_h = history[-2]
            x2_h, f2_h = history[-1]
            
            # Fit quadratic: x = A*f^2 + B*f + C
            # Solve linear system:
            # A*f0^2 + B*f0 + C = x0
            # A*f1^2 + B*f1 + C = x1
            # A*f2^2 + B*f2 + C = x2
            
            # Using Lagrange interpolation for x as a function of f
            # x_new = x0 * (f - f1)(f - f2) / ((f0 - f1)(f0 - f2)) + ...
            # We want x where f=0.
            
            denom1 = (f0_h - f1_h) * (f0_h - f2_h)
            denom2 = (f1_h - f0_h) * (f1_h - f2_h)
            denom3 = (f2_h - f0_h) * (f2_h - f1_h)
            
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                term1 = x0_h * (0 - f1_h) * (0 - f2_h) / denom1
                term2 = x1_h * (0 - f0_h) * (0 - f2_h) / denom2
                term3 = x2_h * (0 - f0_h) * (0 - f1_h) / denom3
                x_new = term1 + term2 + term3
            else:
                x_new = None
        
        # Strategy 2: Secant Method
        if x_new is None and len(history) >= 2:
            x1_h, f1_h = history[-2]
            x2_h, f2_h = history[-1]
            if abs(f1_h - f2_h) > 1e-30:
                x_new = x2_h - f2_h * (x2_h - x1_h) / (f2_h - f1_h)
            else:
                x_new = None
        
        # Strategy 3: Finite Difference Newton
        if x_new is None:
            # Estimate derivative at current_x
            h = 1e-6 * max(1.0, abs(current_x))
            try:
                fx_plus = f(current_x + h)
                if math.isfinite(fx_plus):
                    d = (fx_plus - current_fx) / h
                    if abs(d) > 1e-12:
                        x_new = current_x - current_fx / d
                    else:
                        x_new = current_x + 1e-4 # Small step if derivative is zero
                else:
                    x_new = current_x + 1e-4
            except Exception:
                x_new = current_x + 1e-4

        # Safeguarding
        if x_new is None:
            x_new = current_x + 1e-4
            
        if not math.isfinite(x_new):
            x_new = current_x + 1e-4

        # Check if step is too large
        # Limit step to 10 * max(1, |x|)
        max_step = 10.0 * max(1.0, abs(current_x))
        if abs(x_new - current_x) > max_step:
            x_new = current_x + math.copysign(max_step, x_new - current_x)

        # Evaluate new point
        try:
            fx_new = f(x_new)
        except Exception:
            # If it fails, try a smaller step
            x_new = current_x + 0.1 * (x_new - current_x)
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break

        if not math.isfinite(fx_new):
            # Try smaller step
            x_new = current_x + 0.1 * (x_new - current_x)
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break

        # Accept if finite
        if abs(fx_new) < best_abs_fx:
            best_x, best_fx, best_abs_fx = x_new, fx_new, abs(fx_new)
            
        if abs(fx_new) <= 1e-8:
            return x_new

        # Update history
        # Keep only the last 3 points to avoid memory issues and keep interpolation local
        history.append((x_new, fx_new))
        if len(history) > 3:
            history.pop(0)

    # Return best found
    if abs(best_fx) <= 1e-8:
        return best_x
        
    return best_x
# EVOLVE-BLOCK-END
