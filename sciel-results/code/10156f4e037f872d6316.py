import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 115.0  # safety margin before 120s
    x = float(x0)
    
    # Track best result
    fx = f(x)
    best_x = x
    best_fx = abs(fx)
    
    if best_fx <= 1e-10:
        return x
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Compute derivative using 5-point stencil for better accuracy
        h = 1e-5 * max(1.0, abs(x))
        # 5-point central difference: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        fp = (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12.0 * h)
        
        if abs(fp) < 1e-14:
            # Derivative too small, try a small perturbation
            x = x + 0.1 * max(1.0, abs(x))
            fx = f(x)
            if abs(fx) < best_fx:
                best_x = x
                best_fx = abs(fx)
            continue
        
        # Newton step
        step = fx / fp
        x_new = x - step
        
        # Safeguard: if step is too large, damp it
        if abs(step) > 10.0 * max(1.0, abs(x)):
            step = 10.0 * max(1.0, abs(x)) * (1.0 if step > 0 else -1.0)
            x_new = x - step
        
        fx_new = f(x_new)
        
        if abs(fx_new) < best_fx:
            best_x = x_new
            best_fx = abs(fx_new)
        
        if best_fx <= 1e-10:
            return best_x
        
        # If Newton step made things worse, try damped steps
        if abs(fx_new) > abs(fx):
            # Try halving the step repeatedly
            accepted = False
            for damp in [0.5, 0.25, 0.125, 0.0625]:
                x_damped = x - damp * (fx / fp)
                fx_damped = f(x_damped)
                if abs(fx_damped) < abs(fx):
                    x = x_damped
                    fx = fx_damped
                    accepted = True
                    if abs(fx) < best_fx:
                        best_x = x
                        best_fx = abs(fx)
                    break
            if not accepted:
                # Try moving in the opposite direction
                x_alt = x + 0.1 * max(1.0, abs(x))
                fx_alt = f(x_alt)
                if abs(fx_alt) < abs(fx):
                    x = x_alt
                    fx = fx_alt
                    if abs(fx) < best_fx:
                        best_x = x
                        best_fx = abs(fx)
                else:
                    x = x_new
                    fx = fx_new
        else:
            x = x_new
            fx = fx_new
    
    # Final polish: a few more Newton iterations with tighter tolerance
    for _ in range(10):
        if time.time() > deadline:
            break
        h = 1e-6 * max(1.0, abs(best_x))
        fp = (f(best_x + h) - f(best_x - h)) / (2.0 * h)
        if abs(fp) < 1e-14:
            break
        fx_best = f(best_x)
        x_new = best_x - fx_best / fp
        fx_new = f(x_new)
        if abs(fx_new) < best_fx:
            best_x = x_new
            best_fx = abs(fx_new)
        if best_fx <= 1e-12:
            break
    
    return best_x
# EVOLVE-BLOCK-END
