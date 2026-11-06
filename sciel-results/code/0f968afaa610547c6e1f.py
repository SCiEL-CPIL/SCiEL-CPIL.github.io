"""Root finding: Ill-conditioned - Robust multi-strategy with adaptive refinement."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Strategy:
    1. Quick check at x0 and neighbors
    2. Secant method for rapid convergence
    3. Adaptive Newton with 5-point FD derivative
    4. Bisection/Brent fallback with bracket finding
    5. Final high-precision polish
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin for 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    STRICT_TOL = 1e-10  # Stricter target for polish

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]
    history_x = []
    history_f = []

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() - start_time > MAX_WALL:
            raise RuntimeError("Time limit exceeded")
        
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        
        history_x.append(float(x))
        history_f.append(float(val))
        return val

    def is_good(x, fx, tol=TARGET_TOL):
        return abs(fx) <= tol

    def check_budget():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    def elapsed():
        return time.time() - start_time

    # --- Strategy 1: Secant Method ---
    def try_secant(x0, max_iter=40):
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        # Initialize second point with adaptive step
        h = 1e-3 * max(1.0, abs(x))
        x_prev = x + h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev

        for i in range(max_iter):
            if not check_budget():
                break
            if fx == fx_prev:
                break
            if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
                break
                
            # Secant step
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            # If we're not making progress, try a different step
            if abs(fx_new) >= abs(fx) and abs(fx_new) >= abs(fx_prev):
                # Damp the step
                x_new = x - 0.5 * (fx / d)
                if not math.isfinite(x_new):
                    break
                fx_new = eval_f(x_new)
                if is_good(x_new, fx_new):
                    return x_new
            
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break
        
        return x if abs(fx) <= TARGET_TOL else None

    # --- Strategy 2: Adaptive Newton with 5-point FD ---
    def try_newton(x0, max_iter=30):
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        for i in range(max_iter):
            if not check_budget():
                break
            
            # Adaptive h for 5-point central difference
            # Optimal h ~ (eps * |x|)^(1/4) for 5-point formula
            base_h = 1e-4 * max(1.0, abs(x))
            h = base_h
            if h < 1e-12:
                h = 1e-12
            
            # Evaluate neighbors for derivative
            fx_m2 = eval_f(x - 2*h)
            if is_good(x - 2*h, fx_m2): return x - 2*h
            fx_m1 = eval_f(x - h)
            if is_good(x - h, fx_m1): return x - h
            fx_p1 = eval_f(x + h)
            if is_good(x + h, fx_p1): return x + h
            fx_p2 = eval_f(x + 2*h)
            if is_good(x + 2*h, fx_p2): return x + 2*h
            
            d = (-fx_p2 + 8*fx_p1 - 8*fx_m1 + fx_m2) / (12*h)
            
            if d == 0 or not math.isfinite(d):
                # Try smaller h
                h *= 0.1
                if h < 1e-14:
                    break
                continue
                
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            # Limit step to prevent divergence
            max_step = 0.5 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)
                
            x_new = x - dx
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
                
            # Damping: if we got worse, try halving the step
            if abs(fx_new) > abs(fx):
                improved = False
                for damp in [0.5, 0.25, 0.125, 0.0625]:
                    if not check_budget(): break
                    x_d = x - damp * dx
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        improved = True
                        break
                if not improved:
                    # Try inverse step direction
                    x_d = x + 0.5 * dx
                    if math.isfinite(x_d):
                        fx_d = eval_f(x_d)
                        if is_good(x_d, fx_d):
                            return x_d
                        if abs(fx_d) < abs(fx):
                            x, fx = x_d, fx_d
                    else:
                        break
            else:
                x, fx = x_new, fx_new
                
            if abs(dx) < 1e-13 * max(1.0, abs(x)):
                break

        return x if abs(fx) <= TARGET_TOL else None

    # --- Strategy 3: Bisection with bracket finding ---
    def find_bracket(x0, max_scan=50):
        """Find a bracket [a,b] where f(a)*f(b) < 0"""
        if not check_budget():
            return None
        
        # First check history for existing bracket
        for i in range(len(history_x) - 1):
            if history_f[i] * history_f[i+1] < 0:
                return (history_x[i], history_x[i+1], history_f[i], history_f[i+1])
        
        # Scan around x0 to find a bracket
        base_x = float(x0)
        fa = eval_f(base_x)
        if is_good(base_x, fa):
            return (base_x, base_x, fa, fa)
        
        # Try expanding in both directions
        h = 1e-2 * max(1.0, abs(base_x))
        
        # Scan right
        prev_x = base_x
        prev_fx = fa
        for i in range(1, max_scan // 2 + 1):
            if not check_budget(): break
            cur_x = base_x + i * h
            cur_fx = eval_f(cur_x)
            if is_good(cur_x, cur_fx):
                return (cur_x, cur_x, cur_fx, cur_fx)
            if prev_fx * cur_fx < 0:
                return (prev_x, cur_x, prev_fx, cur_fx)
            prev_x, prev_fx = cur_x, cur_fx
        
        # Scan left
        prev_x = base_x
        prev_fx = fa
        for i in range(1, max_scan // 2 + 1):
            if not check_budget(): break
            cur_x = base_x - i * h
            cur_fx = eval_f(cur_x)
            if is_good(cur_x, cur_fx):
                return (cur_x, cur_x, cur_fx, cur_fx)
            if prev_fx * cur_fx < 0:
                return (cur_x, prev_x, cur_fx, prev_fx)
            prev_x, prev_fx = cur_x, cur_fx
            
        return None

    def try_bisection(a, b, fa, fb, max_iter=60):
        """Bisection method - robust but slower"""
        if not check_budget():
            return None
        
        if fa * fb > 0:
            return None
            
        for i in range(max_iter):
            if not check_budget():
                break
                c = 0.5 * (a + b)
                fc = eval_f(c)
                if is_good(c, fc):
                    return c
                if fa * fc <= 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    
                if abs(b - a) < 1e-14 * max(1.0, abs(c)):
                    break
        
        c = 0.5 * (a + b)
        fc = eval_f(c)
        return c if is_good(c, fc) else None

    def try_brent():
        """Use scipy's brentq if bracket available"""
        if not check_budget():
            return None
            
        bracket = find_bracket(best_x[0])
        if bracket is None:
            return None
            
        a, b, fa, fb = bracket
        if abs(a - b) < 1e-15:
            return a if is_good(a, fa) else b if is_good(b, fb) else None
            
        try:
            from scipy.optimize import brentq
            def f_wrap(x):
                return eval_f(x)
            root = brentq(f_wrap, a, b, xtol=1e-14, rtol=1e-14, maxiter=200)
            fr = eval_f(root)
            if is_good(root, fr):
                return root
        except Exception:
            # Fall back to bisection
            return try_bisection(a, b, fa, fb)
        return None

    # --- Strategy 4: High-precision polish ---
    def polish_root(x_start, fx_start=None, max_iter=25):
        """Final high-precision Newton polish to drive |f(x)| below 1e-8 reliably."""
        if not check_budget():
            return None
        x = float(x_start)
        if fx_start is not None:
            fx = float(fx_start)
        else:
            fx = eval_f(x)
            if is_good(x, fx):
                return x
        
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(fx) <= TARGET_TOL:
                return x
            
            # Use smaller h for higher precision
            h = 1e-5 * max(1.0, abs(x))
            if h < 1e-13:
                h = 1e-13
            
            fx_p = eval_f(x + h)
            if is_good(x + h, fx_p):
                return x + h
            fx_m = eval_f(x - h)
            if is_good(x - h, fx_m):
                return x - h
            
            # Central difference for derivative
            d = (fx_p - fx_m) / (2*h)
            
            if d == 0 or not math.isfinite(d):
                # Try 5-point
                h2 = h * 2
                fx_p2 = eval_f(x + h2)
                if is_good(x + h2, fx_p2): return x + h2
                fx_m2 = eval_f(x - h2)
                if is_good(x - h2, fx_m2): return x - h2
                d = (-fx_p2 + 8*fx_p - 8*fx_m + fx_m2) / (12*h2)
                
                if d == 0 or not math.isfinite(d):
                    break
            
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            x_new = x - dx
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            # Only accept if we made progress
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try damped step
                x_d = x - 0.5 * dx
                if math.isfinite(x_d):
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                    else:
                        break
                else:
                    break
            
            if abs(dx) < 1e-14 * max(1.0, abs(x)):
                break

        return x if abs(fx) <= TARGET_TOL else None

    # === MAIN EXECUTION ===
    
    # 1. Quick check at x0
    fx0 = eval_f(float(x0))
    if is_good(x0, fx0):
        return float(x0)
    
    # 2. Try Secant (efficient, robust for many cases)
    result = try_secant(x0)
    if result is not None:
        return result

    # 3. Try Newton from best point found so far
    result = try_newton(best_x[0])
    if result is not None:
        return result
    
    # 4. Try Newton from original x0 if best_x is far
    if abs(best_x[0] - float(x0)) > 1e-3 * max(1.0, abs(x0)):
        result = try_newton(float(x0))
        if result is not None:
            return result
        
    # 5. Try Brent/Bisection
    result = try_brent()
    if result is not None:
        return result
    
    # 6. Final high-precision polish from best point
    result = polish_root(best_x[0])
    if result is not None:
        return result
    
    # 7. Try polish from x0 as well
    if abs(best_x[0] - float(x0)) > 1e-6:
        result = polish_root(float(x0))
        if result is not None:
            return result
    
    # 8. Final fallback: return best found
    return best_x[0]

# EVOLVE-BLOCK-END
