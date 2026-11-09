"""Root finding for ill-conditioned equations: MODEL-AND-OPTIMIZE with adaptive refinement."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit

    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    # Store all evaluated points for potential re-use or as candidates
    eval_history = []  # list of (x, fval)

    try:
        best_f = f(best_x)
        eval_count += 1
        if math.isfinite(best_f):
            best_abs = abs(best_f)
            eval_history.append((best_x, best_f))
    except Exception:
        best_f = float('inf')
        best_abs = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs:
                best_f = val
                best_x = x
                best_abs = abs_val
            eval_history.append((x, val))
            return val
        except Exception:
            return None

    # 1. Initial Check
    if best_f is not None and best_abs <= 1e-8:
        return best_x

    # Final polish: high-precision Newton/secant refinement
    def polish(x_start, max_iter=50):
        """High-precision polish using secant/Newton with tight tolerance."""
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        
        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
        
        if abs(fx) <= 1e-8:
            return x
        
        # Get a second point for secant
        h = 1e-8 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        if fx2 is None:
            return None
        
        x1, f1 = x, fx
        x0_, f0_ = x2, fx2
        
        for _ in range(max_iter):
            if check_budget():
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_
            
            # Try Newton if we have a good derivative estimate
            if f1 != f0_ and abs(f1 - f0_) > 1e-15:
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
            else:
                # Fallback: small step
                x_new = x1 + 1e-10 * max(1.0, abs(x1))
            
            # Safeguard
            max_jump = max(1e-6, 0.1 * abs(x1))
            if abs(x_new - x1) > max_jump:
                x_new = x1 + math.copysign(max_jump, x_new - x1)
            
            f_new = safe_eval(x_new)
            if f_new is None:
                break
            
            x0_, f0_ = x1, f1
            x1, f1 = x_new, f_new
            
            if abs(x1 - x0_) < 1e-15 * max(1.0, abs(x1)):
                break
        
        return x1 if abs(f1) <= 1e-8 else (x1 if abs(f1) < best_abs else None)

    # Derivative-free local minimization of g(x) = f(x)^2
    def minimize_residual(x_start, max_iter=30):
        """Nelder-Mead style local minimization of f(x)^2."""
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        
        # Initialize simplex
        scale = max(1e-3, 0.01 * abs(x_start))
        x1 = float(x_start)
        x2 = x_start + scale
        x3 = x_start - scale
        
        g1 = safe_eval(x1)
        g2 = safe_eval(x2)
        g3 = safe_eval(x3)
        
        if g1 is None or g2 is None or g3 is None:
            return None
        
        # Sort by g(x) = f(x)^2
        def gval(fv):
            return fv * fv if fv is not None else float('inf')
        
        points = [(x1, g1), (x2, g2), (x3, g3)]
        points.sort(key=lambda p: gval(p[1]) if p[1] is not None else float('inf'))
        
        for _ in range(max_iter):
            if check_budget():
                break
            
            # Check if we found a root
            for _, fv in points:
                if fv is not None and abs(fv) <= 1e-8:
                    return True
            
            # Get the three points sorted by g value
            p_worst = points[-1]  # worst (largest g)
            p_best = points[0]    # best (smallest g)
            p_mid = points[1]     # middle
            
            x_w, f_w = p_worst
            x_b, f_b = p_best
            x_m, f_m = p_mid
            
            # Centroid of best two
            x_c = (x_b + x_m) / 2.0
            
            # Reflection
            x_r = x_c + (x_c - x_w)
            f_r = safe_eval(x_r)
            if f_r is None:
                break
            
            if gval(f_r) < gval(f_m):
                # Expansion
                x_e = x_c + 2.0 * (x_c - x_w)
                f_e = safe_eval(x_e)
                if f_e is not None:
                    if gval(f_e) < gval(f_r):
                        points[-1] = (x_e, f_e)
                    else:
                        points[-1] = (x_r, f_r)
                else:
                    points[-1] = (x_r, f_r)
            elif gval(f_r) < gval(f_w):
                points[-1] = (x_r, f_r)
            else:
                # Contraction
                x_con = x_c - 0.5 * (x_w - x_c)
                f_con = safe_eval(x_con)
                if f_con is not None and gval(f_con) < gval(f_w):
                    points[-1] = (x_con, f_con)
                else:
                    # Shrink
                    for i in range(3):
                        points[i] = (x_b + 0.5 * (points[i][0] - x_b), 
                                     safe_eval(x_b + 0.5 * (points[i][0] - x_b)))
                    # Re-sort
                    points.sort(key=lambda p: gval(p[1]) if p[1] is not None else float('inf'))
            
            # Re-sort
            points.sort(key=lambda p: gval(p[1]) if p[1] is not None else float('inf'))
            
            # Check convergence
            spread = max(abs(p[0] - x_b) for p in points)
            if spread < 1e-12 * max(1.0, abs(x_b)):
                break
        
        return True if best_abs <= 1e-8 else False

    # Helper: Find a root starting from x_start using a hybrid method
    def solve_local(x_start, max_steps=20):
        nonlocal eval_count, best_x, best_f, best_abs
        
        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
            
        if abs(fx) <= 1e-8:
            return x

        # Initial step: establish a second point
        h = 1e-5 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        
        if fx2 is None:
            return x
            
        if abs(fx2) < abs(fx):
            x, fx = x2, fx2
            x2 = x - h
            fx2 = safe_eval(x2)
            if fx2 is None:
                return x
            if abs(fx2) < abs(fx):
                 x, fx = x2, fx2
        
        if x == x2:
             x2 = x + 1e-4
             fx2 = safe_eval(x2)
             if fx2 is None:
                 return x

        x1, f1 = x, fx
        x0_, f0_ = x2, fx2
        
        for _ in range(max_steps):
            if check_budget():
                break
                
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_

            if f1 * f0_ < 0:
                # Bracketed: Bisection
                x_new = (x0_ + x1) / 2.0
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                if f1 * f_new < 0:
                    x0_, f0_ = x1, f1
                    x1, f1 = x_new, f_new
                else:
                    x0_, f0_ = x_new, f_new
                if abs(x1 - x0_) < 1e-12 * max(1.0, abs(x1)):
                    break
            else:
                # Unbracketed: Secant
                if f1 == f0_:
                    x1 = x1 + 1e-4
                    f1 = safe_eval(x1)
                    if f1 is None: break
                    continue
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
                max_jump = max(1.0, 0.5 * abs(x1))
                if abs(x_new - x1) > max_jump:
                    x_new = x1 + math.copysign(max_jump, x_new - x1)
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                x0_, f0_ = x1, f1
                x1, f1 = x_new, f_new
                if abs(x1 - x0_) < 1e-14:
                    break
        
        # If close but not quite there, try polish
        if best_abs < 1e-4 and best_abs > 1e-8:
            result = polish(best_x, max_iter=40)
            if result is not None and best_abs <= 1e-8:
                return result
        
        return x1

    # Main Loop
    # Strategy 1: Direct local solve from x0
    result = solve_local(x0, max_steps=30)
    if best_abs <= 1e-8:
        return best_x

    # Strategy 2: Perturbed starts
    perturbations = [
        1e-2, -1e-2,
        1e-3, -1e-3,
        1e-1, -1e-1,
        0.5, -0.5,
        1.0, -1.0
    ]
    
    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        result = solve_local(x_pert, max_steps=15)
        if best_abs <= 1e-8:
            return best_x

    # Strategy 3: Broader sampling
    radii = [10.0, 100.0, 1000.0]
    for r in radii:
        if check_budget():
            break
        samples = [
            x0 - r,
            x0 - r/2,
            x0 + r/2,
            x0 + r,
            x0 - r*0.1,
            x0 + r*0.1
        ]
        
        for x_s in samples:
            if check_budget():
                break
            fs = safe_eval(x_s)
            if fs is not None:
                if best_abs <= 1e-8:
                    return best_x
                result = solve_local(x_s, max_steps=10)
                if best_abs <= 1e-8:
                    return best_x

    # Strategy 4: Derivative-free minimization of f(x)^2
    if best_abs < 1.0:  # Only try if we're reasonably close
        minimize_residual(best_x, max_iter=30)
        if best_abs <= 1e-8:
            return best_x

    # Strategy 5: Final polish attempt
    if best_abs < 1e-4:
        result = polish(best_x, max_iter=50)
        if result is not None:
            return best_x

    return best_x

# EVOLVE-BLOCK-END
