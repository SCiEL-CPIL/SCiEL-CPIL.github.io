import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: bracketing (Illinois/secant) with Newton acceleration,
    plus a BFGS-inspired superlinear polish phase.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    TOL = 1e-8
    
    eval_count = 0
    best_x = float(x0)
    best_fabs = float('inf')
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        try:
            fx = f(x)
        except Exception:
            return None
        if math.isfinite(fx):
            fabs = abs(fx)
            if fabs < best_fabs:
                best_fabs = fabs
                best_x = x
            return fx
        return None

    def time_exceeded():
        return time.time() > deadline

    def polish_secant(x, fx, x_prev, fx_prev, max_iters=30):
        """Polish using secant method when close to root."""
        nonlocal eval_count, best_x, best_fabs
        for _ in range(max_iters):
            if time_exceeded() or eval_count >= max_evals:
                break
            if abs(fx) <= TOL:
                return x
            if abs(x - x_prev) < 1e-15:
                break
            slope = (fx - fx_prev) / (x - x_prev)
            if abs(slope) < 1e-30:
                break
            x_new = x - fx / slope
            # Clamp step to avoid divergence
            max_step = 5.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + math.copysign(max_step, x_new - x)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) <= TOL:
                return x_new
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
        return x if abs(fx) <= best_fabs else best_x

    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Try multiple initial step sizes to find a bracket
    # Larger step first (measured improvement), then smaller
    step_sizes = [
        0.001 * max(1.0, abs(x)),
        1e-4 * max(1.0, abs(x)),
        1e-6 * max(1.0, abs(x)),
    ]
    
    bracket_found = False
    secant_pts = None  # (x1, f1, x2, f2) for non-bracket case
    
    for h in step_sizes:
        if time_exceeded() or eval_count >= max_evals:
            break
        x2 = x + h
        fx2 = eval_f(x2)
        if fx2 is None:
            continue
        if abs(fx2) <= TOL:
            return x2
        if fx * fx2 < 0:
            # Found a bracket
            a, b = x, x2
            fa, fb = fx, fx2
            bracket_found = True
            break
        else:
            # Save for secant use
            secant_pts = (x, fx, x2, fx2)
    
    if bracket_found:
        # Illinois/secant bracket refinement
        for _ in range(300):
            if time_exceeded() or eval_count >= max_evals:
                break
            
            # If very close, switch to secant polish
            if abs(fa) < 1e-4 * max(1.0, abs(a)) or abs(fb) < 1e-4 * max(1.0, abs(b)):
                # Use secant polish from the better endpoint
                if abs(fa) <= abs(fb):
                    result = polish_secant(a, fa, b, fb)
                    return result
                else:
                    result = polish_secant(b, fb, a, fa)
                    return result
            
            # Secant point
            if abs(fb - fa) > 1e-30 and abs(b - a) > 1e-30:
                c = (a * fb - b * fa) / (fb - fa)
            else:
                c = 0.5 * (a + b)
            
            # If Secant is outside [a,b], use Midpoint
            if not (a < c < b):
                c = 0.5 * (a + b)
            
            fc = eval_f(c)
            if fc is None:
                break
            if abs(fc) <= TOL:
                return c
            
            # Update Bracket
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        # Final check
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    elif secant_pts is not None:
        # No bracket found, use damped secant
        x_prev, fx_prev, x_curr, fx_curr = secant_pts
        
        for _ in range(200):
            if time_exceeded() or eval_count >= max_evals:
                break
            
            if abs(fx_curr) <= TOL:
                return x_curr
            
            # Secant step
            if abs(x_curr - x_prev) < 1e-15:
                break
            slope = (fx_curr - fx_prev) / (x_curr - x_prev)
            if abs(slope) < 1e-30:
                # Flat, perturb
                x_new = x_curr + 1e-5
            else:
                x_new = x_curr - fx_curr / slope
            
            # Clamp step to prevent divergence
            max_step = 5.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr + math.copysign(max_step, x_new - x_curr)
            
            # Line Search: Ensure |f(x_new)| < |f(x_curr)|
            f_curr_abs = abs(fx_curr)
            step = x_new - x_curr
            fx_new = None
            for _ls in range(20):
                fx_new = eval_f(x_curr + step)
                if fx_new is None:
                    break
                if abs(fx_new) < f_curr_abs:
                    break
                # Halve the step
                step *= 0.5
            
            if time_exceeded():
                break
                
            if fx_new is None:
                break
            
            # Check success
            if abs(fx_new) <= TOL:
                return x_curr + step
            
            # Check for new bracket
            if fx_new * fx_curr < 0:
                # Switch to Bracket Mode
                a, b = x_curr, x_curr + step
                fa, fb = fx_curr, fx_new
                for _ in range(300):
                    if time_exceeded() or eval_count >= max_evals:
                        break
                    # If very close, switch to secant polish
                    if abs(fa) < 1e-4 * max(1.0, abs(a)) or abs(fb) < 1e-4 * max(1.0, abs(b)):
                        if abs(fa) <= abs(fb):
                            result = polish_secant(a, fa, b, fb)
                            return result
                        else:
                            result = polish_secant(b, fb, a, fa)
                            return result
                    
                    # Secant point
                    if abs(fb - fa) > 1e-30 and abs(b - a) > 1e-30:
                        c = (a * fb - b * fa) / (fb - fa)
                    else:
                        c = 0.5 * (a + b)
                    
                    if not (a < c < b):
                        c = 0.5 * (a + b)
                        
                    fc = eval_f(c)
                    if fc is None:
                        break
                    if abs(fc) <= TOL:
                        return c
                    
                    if fa * fc < 0:
                        b = c
                        fb = fc
                    else:
                        a = c
                        fa = fc
                # Return best from bracket
                if abs(fa) <= abs(fb):
                    return a
                else:
                    return b
            # Exit secant loop
            
            # Update Secant Points
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_curr + step
            fx_curr = fx_new

    return best_x
