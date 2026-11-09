"""Root finding for ill-conditioned equations: Nelder-Mead on f(x)^2 with adaptive restarts."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit
    
    eval_count = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def safe_eval(x):
        nonlocal eval_count, best_x, best_abs_f
        if eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = x
            return val
        except Exception:
            return None

    def check_budget():
        return eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE

    # 1. Initial Check
    f0 = safe_eval(best_x)
    if f0 is not None and abs(f0) <= 1e-8:
        return best_x

    # 2. Nelder-Mead Simplex Method on g(x) = f(x)^2
    # In 1D, the simplex has 2 vertices.
    # We use a robust implementation with reflection, expansion, contraction, and shrink.
    # We also implement adaptive restarts if the simplex becomes too small or stuck.
    
    # Initialize simplex
    scale = max(1.0, abs(float(x0)))
    delta = 1e-3 * scale
    
    # Vertices: (x, g(x))
    # g(x) = f(x)^2
    def g(x):
        fx = safe_eval(x)
        if fx is None:
            return None
        return fx * fx

    x1 = float(x0)
    x2 = x1 + delta
    
    g1 = g(x1)
    g2 = g(x2)
    
    if g1 is None or g2 is None:
        # If initial points fail, try a different delta
        delta = 1e-2 * scale
        x2 = x1 + delta
        g2 = g(x2)
        if g2 is None:
            return best_x

    # Ensure g1 <= g2
    if g1 > g2:
        x1, x2 = x2, x1
        g1, g2 = g2, g1

    # Nelder-Mead parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink

    # We will run Nelder-Mead for a certain number of iterations or until convergence
    # If it gets stuck, we restart with a larger delta centered at the best point found.
    
    max_nm_iters = 200
    restart_count = 0
    max_restarts = 5

    for restart in range(max_restarts):
        # Reset simplex if restarting
        if restart > 0:
            # Center at best_x
            x1 = best_x
            # Choose a new delta, larger than before
            delta = (1e-3 * (10 ** restart)) * scale
            if delta < 1e-12:
                delta = 1e-12
            if delta > 1e6 * scale:
                delta = 1e6 * scale
            x2 = x1 + delta
            
            g1 = g(x1)
            g2 = g(x2)
            
            if g1 is None or g2 is None:
                break
                
            if g1 > g2:
                x1, x2 = x2, x1
                g1, g2 = g2, g1

        # Run Nelder-Mead
        for i in range(max_nm_iters):
            if check_budget():
                return best_x
            
            # Check convergence
            if abs(g1) <= 1e-16:  # g(x) = f(x)^2, so f(x) <= 1e-8
                return best_x
            
            # Check if simplex is small
            if abs(x2 - x1) < 1e-12 * max(1.0, abs(x1)):
                break
            
            # Reflection
            # Midpoint
            xm = 0.5 * (x1 + x2)
            # Reflected point: x_r = xm + alpha * (xm - x2) = xm + alpha * (x1 - xm)
            # Since x1 is better (lower g), we reflect x2 across xm towards x1 and beyond.
            # x_r = x1 + alpha * (x1 - x2) ? No.
            # Standard NM in 1D:
            # If x1 is best, x2 is worst.
            # x_r = x1 + alpha * (x1 - x2) is not correct.
            # Let's use the standard formula: x_r = 2*xm - x2 + alpha*(xm - x2) ?
            # Actually, for 1D, reflection of x2 about x1 (if x1 is best) is x_r = x1 + alpha*(x1 - x2).
            # Let's stick to the standard: reflect worst point about the centroid of the other points.
            # In 1D, centroid of {x1} is x1.
            # So x_r = x1 + alpha * (x1 - x2).
            
            x_r = x1 + alpha * (x1 - x2)
            g_r = g(x_r)
            
            if g_r is None:
                break
                
            if g_r < g1:
                # Try expansion
                x_e = x1 + gamma * (x_r - x1)
                g_e = g(x_e)
                
                if g_e is not None and g_e < g_r:
                    x1, x2 = x_e, x_r
                    g1, g2 = g_e, g_r
                else:
                    x1, x2 = x_r, x2
                    g1, g2 = g_r, g2
            else:
                if g_r < g2:
                    # Accept reflection
                    x2 = x_r
                    g2 = g_r
                else:
                    # Contraction
                    # Inside contraction (towards best)
                    x_c = x1 + rho * (x2 - x1)
                    g_c = g(x_c)
                    
                    if g_c is not None and g_c < g2:
                        x2 = x_c
                        g2 = g_c
                    else:
                        # Shrink
                        x2 = x1 + sigma * (x2 - x1)
                        g2 = g(x2)
                        
                        if g2 is None:
                            break
                        
                        # If shrink is too small, restart
                        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x1)):
                            break

        # Check if we found a root
        if best_abs_f <= 1e-8:
            return best_x

        # If we broke out of the loop without finding a root, restart with larger delta
        # If simplex is too small, restart
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x1)):
            continue
            
        # If no improvement in this round, restart
        # We can check if g1 has changed significantly from the start of this restart
        # For simplicity, just restart if we used up the iterations or broke out.

    # 3. Final Polish
    # If Nelder-Mead didn't reach 1e-8, try a local secant or bisection polish around best_x
    # This is a hybrid approach.
    
    x_center = best_x
    f_center = safe_eval(x_center)
    
    if f_center is None:
        return best_x
        
    if abs(f_center) <= 1e-8:
        return best_x
        
    # Try to find a bracket
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    # Small steps
    step_sizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    for step in step_sizes:
        if check_budget():
            break
            
        x_left = x_center - step
        x_right = x_center + step
        
        fl = safe_eval(x_left)
        fr = safe_eval(x_right)
        
        if fl is None or fr is None:
            continue
            
        # Check sign change
        if fl * fr < 0:
            bracket_a = x_left
            bracket_b = x_right
            fa = fl
            fb = fr
            break

    # If we have a bracket, use Bisection/Secant hybrid
    if bracket_a is not None and bracket_b is not None:
        a, b = bracket_a, bracket_b
        fa_val, fb_val = fa, fb
        
        for _ in range(50):
            if check_budget():
                break
                
            # Check convergence
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                x_mid = 0.5 * (a + b)
                f_mid = safe_eval(x_mid)
                if f_mid is not None:
                    return x_mid
                break
                
            # Secant step
            if fa_val != fb_val:
                x_new = b - fb_val * (b - a) / (fb_val - fa_val)
            else:
                x_new = 0.5 * (a + b)
                
            # If x_new is outside bracket, use bisection
            if x_new <= a or x_new >= b:
                x_new = 0.5 * (a + b)
                
            f_new = safe_eval(x_new)
            if f_new is None:
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            # Update bracket
            if fa_val * f_new < 0:
                b = x_new
                fb_val = f_new
            else:
                a = x_new
                fa_val = f_new

    # If no bracket found, use Secant from best_x
    step = 1e-6 * max(1.0, abs(best_x))
    x2 = best_x + step
    f2 = safe_eval(x2)
    
    if f2 is not None:
        x1, f1 = best_x, safe_eval(best_x)
        
        if f1 is not None:
            for _ in range(50):
                if check_budget():
                    break
                if f1 == f2:
                    break
                x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                
                # Safeguard
                max_jump = max(1.0, 0.5 * abs(x1))
                if abs(x_new - x1) > max_jump:
                    x_new = x1 + math.copysign(max_jump, x_new - x1)
                    
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                    
                if abs(f_new) <= 1e-8:
                    return x_new
                    
                if abs(f_new) < abs(f1):
                    x2, f2 = x1, f1
                    x1, f1 = x_new, f_new
                else:
                    x2, f2 = x_new, f_new
                    
                if abs(x_new - x2) < 1e-14:
                    break

    return best_x

# EVOLVE-BLOCK-END
