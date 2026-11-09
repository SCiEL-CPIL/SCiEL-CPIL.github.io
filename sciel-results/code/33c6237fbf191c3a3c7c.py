"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Finds a root of f using a high-precision one-shot Newton step.
    Designed for smooth polynomial equations where a local linear
    approximation from a high-order finite difference is sufficient
    to locate the root in a single large step.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    x = float(x0)
    call_count = 0
    best_x = x
    best_f_val = float('inf')
    
    def check_time():
        return (time.time() - start_time) > deadline

    def record(x_val, f_val):
        nonlocal best_x, best_f_val, call_count
        if math.isfinite(f_val) and abs(f_val) < abs(best_f_val):
            best_x = x_val
            best_f_val = f_val

    try:
        # 1. Evaluate at x0
        fx0 = f(x)
        call_count += 1
        record(x, fx0)
        
        if abs(fx0) <= 1e-8:
            return x

        # 2. High-order finite difference to estimate derivative
        # Using 4-point stencil for O(h^4) accuracy on derivative.
        # f'(x) ~ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
        # We need f(x-h), f(x+h), f(x-2h), f(x+2h). We already have f(x).
        
        # Adaptive step size h
        # Polynomials are smooth, so a small relative h is safe.
        # If x is very small, use absolute h.
        scale = max(1.0, abs(x))
        h = 1e-5 * scale
        
        # Evaluate neighbors
        x1 = x - h
        x2 = x + h
        x3 = x - 2*h
        x4 = x + 2*h
        
        try:
            fx1 = f(x1)
            call_count += 1
            record(x1, fx1)
            if abs(fx1) <= 1e-8: return x1
            
            fx2 = f(x2)
            call_count += 1
            record(x2, fx2)
            if abs(fx2) <= 1e-8: return x2
            
            fx3 = f(x3)
            call_count += 1
            record(x3, fx3)
            if abs(fx3) <= 1e-8: return x3
            
            fx4 = f(x4)
            call_count += 1
            record(x4, fx4)
            if abs(fx4) <= 1e-8: return x4
        except Exception:
            # If evaluation fails, fall back to best found or standard Newton
            pass

        # Calculate derivative estimate
        # If we have all points
        if call_count >= 5:
            denom = 12.0 * h
            if denom != 0:
                d_est = (fx3 - 8*fx1 + 8*fx2 - fx4) / denom
                if math.isfinite(d_est) and abs(d_est) > 1e-15:
                    # One-shot Newton step
                    x_new = x - fx0 / d_est
                    
                    if math.isfinite(x_new):
                        # Verify the step
                        try:
                            fx_new = f(x_new)
                            call_count += 1
                            record(x_new, fx_new)
                            if abs(fx_new) <= 1e-8:
                                return x_new
                            
                            # If not close enough, do one standard Newton refinement
                            # using the new point and derivative estimate (or re-estimate)
                            # Re-estimate derivative at x_new for better accuracy if needed
                            if call_count < 490 and not check_time():
                                h2 = 1e-5 * max(1.0, abs(x_new))
                                # Simple 2-point derivative for refinement
                                fx_new_h = f(x_new + h2)
                                call_count += 1
                                record(x_new + h2, fx_new_h)
                                
                                d_new = (fx_new_h - fx_new) / h2
                                if math.isfinite(d_new) and abs(d_new) > 1e-15:
                                    x_final = x_new - fx_new / d_new
                                    if math.isfinite(x_final):
                                        fx_final = f(x_final)
                                        call_count += 1
                                        record(x_final, fx_final)
                                        if abs(fx_final) <= 1e-8:
                                            return x_final
                                        # If still not good, return best
                        except Exception:
                            pass

        # Fallback: Standard Newton with FD if one-shot failed
        # This handles cases where the initial step was bad or derivative was zero
        if call_count < 490 and not check_time():
            for _ in range(10):
                if call_count >= 490: break
                try:
                    fx = f(x)
                    call_count += 1
                    record(x, fx)
                    if abs(fx) <= 1e-8:
                        return x
                    
                    h = 1e-6 * max(1.0, abs(x))
                    fx_h = f(x + h)
                    call_count += 1
                    record(x + h, fx_h)
                    
                    d = (fx_h - fx) / h
                    if abs(d) < 1e-15:
                        break
                    x = x - fx / d
                    if not math.isfinite(x):
                        break
                except Exception:
                    break

    except Exception:
        pass

    # Return the best x found
    if best_f_val == float('inf'):
        return x0
    return best_x
# EVOLVE-BLOCK-END
