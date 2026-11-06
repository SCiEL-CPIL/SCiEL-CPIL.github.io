"""Root finding: Ill-conditioned - robust solver with high-precision derivatives and damping."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust Newton-Secant hybrid with high-precision finite differences
    and adaptive damping.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    # Global evaluation counter
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            return 0.0
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_time():
        return time.time() - start_time < MAX_WALL

    # Helper: 5-point central difference for derivative
    def derivative_5pt(x, fx):
        # h ~ eps^(1/4) * max(1, |x|) is optimal for 5-point stencil
        h = (1e-7) ** 0.25 * max(1.0, abs(x))
        # Ensure h is not too small relative to machine precision
        h = max(h, 1e-8)
        
        fx_m2 = eval_f(x - 2*h)
        fx_m1 = eval_f(x - h)
        fx_p1 = eval_f(x + h)
        fx_p2 = eval_f(x + 2*h)
        
        # Check for good points in the neighborhood
        if is_good(x - 2*h, fx_m2): return None, x - 2*h, fx_m2
        if is_good(x - h, fx_m1): return None, x - h, fx_m1
        if is_good(x + h, fx_p1): return None, x + h, fx_p1
        if is_good(x + 2*h, fx_p2): return None, x + 2*h, fx_p2
        
        d = (-fx_p2 + 8*fx_p1 - 8*fx_m1 + fx_m2) / (12*h)
        return d, None, None

    def try_newton_damped(x0, max_iter=40):
        """Newton's method with 5-point derivative and damping."""
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        best_x = x
        best_f = abs(fx)
        
        for i in range(max_iter):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
                
            d, x_good, f_good = derivative_5pt(x, fx)
            if x_good is not None:
                return x_good
            
            if d is None or d == 0 or not math.isfinite(d):
                break
            
            # Newton step
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            # Limit step size to prevent divergence
            # For ill-conditioned problems, steps can be huge.
            max_step = 1.0 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)
            
            x_new = x - dx
            
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            # Damping: if |f| didn't decrease, reduce step
            if abs(fx_new) >= best_f:
                # Try half steps
                for damp in [0.5, 0.25, 0.125]:
                    x_damped = x - damp * (x - x_new) # x - damp*dx
                    # Re-evaluate
                    if eval_count[0] >= MAX_EVALS: break
                    fx_damped = eval_f(x_damped)
                    if is_good(x_damped, fx_damped):
                        return x_damped
                    if abs(fx_damped) < abs(fx):
                        x = x_damped
                        fx = fx_damped
                        if abs(fx) < best_f:
                            best_x = x
                            best_f = abs(fx)
                        break
                else:
                    # If no damping worked, break to avoid oscillation
                    break
            else:
                x = x_new
                fx = fx_new
                if abs(fx) < best_f:
                    best_x = x
                    best_f = abs(fx)
            
            # Convergence check
            if abs(dx) < 1e-10 * max(1.0, abs(x)):
                break
        
        return best_x if best_f <= TARGET_TOL else None

    def try_secant(x0, max_iter=40):
        """Secant method with damping."""
        x = float(x0)
        # Initial second point
        h = 1e-4 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        
        if is_good(x_prev, fx_prev):
            return x_prev
            
        fx = eval_f(x)
        if is_good(x, fx):
            return x
            
        best_x = x if abs(fx) < abs(fx_prev) else x_prev
        best_f = min(abs(fx), abs(fx_prev))
        
        for i in range(max_iter):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
                
            if fx == fx_prev:
                break
                
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
                
            if abs(fx_new) < best_f:
                best_f = abs(fx_new)
                best_x = x_new
                
            # Damping logic similar to Newton
            if abs(fx_new) >= abs(fx):
                # If it got worse, maybe we should stop or damp
                # For secant, damping is tricky. Let's just update and continue, 
                # but break if we oscillate.
                pass
            
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            
            if abs(x_new - x_prev) < 1e-10 * max(1.0, abs(x_new)):
                break
                
        return best_x if best_f <= TARGET_TOL else None

    def try_bisection(x0, max_iter=60):
        """Bisection with robust bracketing."""
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
            
        # Find bracket
        left, right, fl, fr = None, None, None, None
        h = 1e-2 * max(1.0, abs(x))
        
        for scale in [1, 10, 100, 1000, 1e4, 1e5, 1e6]:
            h_try = h * scale
            xl = x - h_try
            xr = x + h_try
            
            fl_try = eval_f(xl)
            if is_good(xl, fl_try): return xl
            
            fr_try = eval_f(xr)
            if is_good(xr, fr_try): return xr
            
            if fl_try * fr_try < 0:
                left, right = xl, xr
                fl, fr = fl_try, fr_try
                break
        
        if left is None:
            return None
            
        # Bisection
        best_x = left if abs(fl) < abs(fr) else right
        best_f = min(abs(fl), abs(fr))
        
        for i in range(max_iter):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
                
            mid = (left + right) / 2
            fm = eval_f(mid)
            if is_good(mid, fm):
                return mid
                
            if abs(fm) < best_f:
                best_f = abs(fm)
                best_x = mid
                
            if fl * fm <= 0:
                right = mid
                fr = fm
            else:
                left = mid
                fl = fm
                
            if abs(right - left) < 1e-12:
                break
                
        return best_x if best_f <= TARGET_TOL else None

    # Strategy 1: Newton with high precision derivative
    result = try_newton_damped(x0)
    if result is not None:
        return result

    # Strategy 2: Secant
    result = try_secant(x0)
    if result is not None:
        return result

    # Strategy 3: Bisection
    result = try_bisection(x0)
    if result is not None:
        return result

    # Strategy 4: Perturbed Newton
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1]:
        if not check_time() or eval_count[0] >= MAX_EVALS:
            break
        x_perturbed = x0 + offset * max(1.0, abs(x0))
        result = try_newton_damped(x_perturbed, max_iter=30)
        if result is not None:
            return result

    # Strategy 5: Perturbed Secant
    for offset in [1e-3, -1e-3, 1e-2, -1e-2]:
        if not check_time() or eval_count[0] >= MAX_EVALS:
            break
        x_perturbed = x0 + offset * max(1.0, abs(x0))
        result = try_secant(x_perturbed, max_iter=30)
        if result is not None:
            return result

    # Strategy 6: Perturbed Bisection
    for offset in [1e-2, -1e-2, 0.1, -0.1]:
        if not check_time() or eval_count[0] >= MAX_EVALS:
            break
        x_perturbed = x0 + offset * max(1.0, abs(x0))
        result = try_bisection(x_perturbed, max_iter=50)
        if result is not None:
            return result

    # Fallback: Return x0 (or best found, but we didn't track global best easily here)
    return float(x0)

# EVOLVE-BLOCK-END
