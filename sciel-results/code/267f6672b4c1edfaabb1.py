"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s

    x = float(x0)

    # Get first evaluation
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_fx = fx
    call_count = 1

    # Try polynomial reconstruction with 4 points (for cubic or lower)
    # Use points: x, x+h, x+2h, x+3h
    h = 1e-4 * max(1.0, abs(x))
    points = [x, x + h, x + 2*h, x + 3*h]
    values = [fx]
    
    for i in range(1, 4):
        if time.time() > deadline:
            return best_x
        try:
            val = f(points[i])
        except Exception:
            break
        call_count += 1
        if not math.isfinite(val):
            break
        if abs(val) < abs(best_fx):
            best_x = points[i]
            best_fx = val
        if abs(val) <= 1e-8:
            return points[i]
        values.append(val)
    
    # If we have 4 valid points, try to reconstruct polynomial and find roots
    if len(values) == 4:
        try:
            # Lagrange interpolation to get polynomial coefficients
            # We want p(x) such that p(points[i]) = values[i]
            # Use numpy.polyfit to get coefficients (highest degree first)
            coeffs = np.polyfit(points, values, 3)
            
            # Find roots of the polynomial
            roots = np.roots(coeffs)
            
            # Filter for real roots that are close to x0
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            
            if real_roots:
                # Pick the root closest to x0
                best_root = min(real_roots, key=lambda r: abs(r - x0))
                
                # Verify the root
                try:
                    f_val = f(best_root)
                except Exception:
                    f_val = float('inf')
                call_count += 1
                
                if math.isfinite(f_val) and abs(f_val) <= 1e-8:
                    return best_root
                elif math.isfinite(f_val) and abs(f_val) < abs(best_fx):
                    best_x = best_root
                    best_fx = f_val
        except Exception:
            pass  # Fall through to secant method

    # If polynomial reconstruction didn't work, fall back to secant method
    # Get second point for secant if we don't already have good points
    if call_count < 2:
        h2 = 1e-4 * max(1.0, abs(x))
        x2 = x + h2
        try:
            fx2 = f(x2)
        except Exception:
            return best_x
        call_count += 1
        if not math.isfinite(fx2):
            return best_x
        if abs(fx2) < abs(best_fx):
            best_x = x2
            best_fx = fx2
        if abs(fx2) <= 1e-8:
            return x2
    else:
        # Use the last two points from our polynomial attempt
        x = points[-2]
        fx = values[-2]
        x2 = points[-1]
        fx2 = values[-1]

    # Secant method iteration
    for i in range(480):  # 2 + 480 = 482 calls max, well under 500
        if time.time() > deadline:
            return best_x

        denom = fx2 - fx
        if abs(denom) < 1e-30:
            # Nearly parallel; try a different step from x2
            h2 = 1e-3 * max(1.0, abs(x2))
            x_new = x2 + h2
            try:
                fx_new = f(x_new)
            except Exception:
                return best_x
            call_count += 1
            if not math.isfinite(fx_new):
                return best_x
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x, fx = x2, fx2
            x2, fx2 = x_new, fx_new
            continue

        x_new = x2 - fx2 * (x2 - x) / denom
        if not math.isfinite(x_new):
            return best_x

        try:
            fx_new = f(x_new)
        except Exception:
            return best_x

        call_count += 1
        if not math.isfinite(fx_new):
            return best_x

        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        if abs(fx_new) <= 1e-8:
            return x_new

        # Shift: x = x2, x2 = x_new
        x, fx = x2, fx2
        x2, fx2 = x_new, fx_new

    return best_x
# EVOLVE-BLOCK-END
