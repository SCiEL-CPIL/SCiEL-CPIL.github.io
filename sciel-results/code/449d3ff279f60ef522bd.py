"""Root finding for ill-conditioned equations: Multi-scale Golden Section + Brent Polish."""
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

    # 2. Multi-scale Global Search
    # We use a logarithmic sweep of intervals centered at x0.
    # For each interval [x0 - R, x0 + R], we perform a coarse Golden Section search
    # to find the minimum of |f(x)|. This avoids relying on local derivatives or 
    # sign changes which can fail for ill-conditioned or tangent roots.
    
    # Parameters for Golden Section
    gr = (math.sqrt(5) - 1) / 2  # 0.618...
    tol_x = 1e-12
    
    # We will try radii from small to large, but also large to small?
    # Usually, the root is near x0. But for ill-conditioned, maybe not.
    # Let's try a few scales: 1e-6, 1e-4, 1e-2, 1, 10, 100, 1000, 1e6
    # We need to be careful with the number of evaluations.
    # Golden section reduces interval length by factor ~0.618 per step.
    # To reduce from R to tol, we need log(R/tol) / log(1/gr) steps.
    # log(1e6/1e-12) / log(1.618) ~ 18 / 0.48 ~ 37 steps.
    # Each step adds 1 eval (after initial 2). So ~40 evals per scale.
    # With 8 scales, that's 320 evals. Leaves 160 for polishing.
    
    scales = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e6, 1e8]
    # Filter scales that are too small or too large relative to x0?
    # x0 is the anchor.
    base_scale = max(1.0, abs(float(x0)))
    
    # Generate scales relative to base_scale to handle x0 magnitude
    rel_scales = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e6, 1e8]
    radii = [base_scale * s for s in rel_scales]
    
    # Sort radii? Maybe start with mid-range?
    # Let's just iterate. If we find a very small |f|, we can stop early.
    
    found_bracket = None
    found_min_x = None
    found_min_val = float('inf')
    
    for R in radii:
        if check_budget():
            break
            
        a = float(x0) - R
        b = float(x0) + R
        
        # Handle potential overflow/underflow in a,b
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
            
        # Golden Section Search on [a, b] to minimize |f(x)|
        # Initial points
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = safe_eval(c)
        fd = safe_eval(d)
        
        if fc is None or fd is None:
            continue
            
        # Update global best
        # safe_eval already updates best_x/best_abs_f
        
        # Iterate
        max_gs_steps = 60  # Cap steps per scale to save budget
        for _ in range(max_gs_steps):
            if check_budget():
                break
                
            # If interval is small enough, stop
            if abs(b - a) < tol_x * max(1.0, abs(a), abs(b)):
                break
                
            if fc < fd:
                # Minimum is in [a, d]
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = safe_eval(c)
                if fc is None:
                    break
            else:
                # Minimum is in [c, b]
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = safe_eval(d)
                if fd is None:
                    break
            
            # Check for root found
            if best_abs_f <= 1e-8:
                return best_x
                
            # Early termination if we've converged well
            if abs(fc - fd) < 1e-10 * max(1.0, abs(fc), abs(fd)):
                break
        
        # Record the best point found in this interval
        # The minimum is likely between c and d, or one of them.
        # We rely on best_x/best_abs_f which is updated in safe_eval.
        # If we found a significantly better point, we might want to refine it specifically.
        
        # If we found a very good point, we can break and polish.
        if best_abs_f < 1e-4:
            # We have a good candidate. Let's try to polish it now.
            # But maybe a larger scale has a better root?
            # For ill-conditioned, local minima of |f| are usually the root.
            # Let's stop global search and polish.
            break

    # 3. Polish with Brent's Method
    # We need a bracket [a, b] where f(a) and f(b) have opposite signs, 
    # OR we can use secant/Newton if we don't have a bracket.
    # Since we minimized |f|, we might not have a sign change (tangent root).
    # However, if we found a minimum of |f| near 0, f is likely changing sign nearby 
    # unless it's a tangent root (f(x) >= 0 or <= 0 everywhere).
    # If it's a tangent root, Brent won't work directly.
    # But the task says "ill-conditioned", which often implies steep gradients, not necessarily tangent.
    # Let's try to find a bracket around best_x.
    
    # If we have a bracket from the last GS step?
    # The GS loop ended with a and b. Let's check if f(a) and f(b) have opposite signs.
    # We need to re-evaluate a and b to be sure, or use cached values?
    # We didn't cache all. Let's just use best_x and try a local bracket search.
    
    # Local Bracket Search
    # Start from best_x, expand radius until we find a sign change.
    # If we find a sign change, use Brent.
    # If we don't, use Secant/Newton hybrid.
    
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
    step_sizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    
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
            
        # If one side is better, maybe shrink? No, just expand.
        # If we find a point with |f| < |f_center|, update best_x?
        # safe_eval already does this.
        
        # If we've expanded too much, stop.
        if step > max(1e6, 1e6 * abs(x_center)):
            break

    # If we have a bracket, use Brent's Method
    if bracket_a is not None and bracket_b is not None:
        # Implement a simplified Brent's method or use SciPy?
        # "Python standard library, NumPy and SciPy are available."
        # Let's use scipy.optimize.brentq for robustness if available.
        try:
            from scipy.optimize import brentq
            # brentq requires f(a)*f(b) < 0
            # It also has maxiter and xtol
            # We need to wrap f to count evals?
            # scipy.brentq will call f multiple times.
            # We can't easily limit its evals to our budget unless we wrap it.
            
            # Let's write a simple Brent-like or just Secant since we have a bracket.
            # Actually, Illinois algorithm is good for ill-conditioned.
            # Or just Bisection + Secant hybrid.
            
            # Let's use a simple robust hybrid:
            # If secant step goes outside bracket, do bisection.
            
            a, b = bracket_a, bracket_b
            fa_val, fb_val = fa, fb
            
            # Ensure fa is smaller magnitude? Not required for bisection.
            # For secant, we use the two most recent points.
            
            # Let's stick to a robust Illinois/Brent hybrid implementation.
            # Since we have a bracket, we can use a modified Illinois method.
            
            for _ in range(50):
                if check_budget():
                    break
                    
                # Check convergence
                if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                    # Return midpoint
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
                    
                # Illinois modification: if we bisected, shrink the larger side more
                # This is complex. Let's just use standard Bisection/Secant.
                
        except ImportError:
            # Fallback to simple bisection if SciPy not available (it should be)
            pass

    # If no bracket found, use Secant/Newton from best_x
    # We need a second point.
    # Use the last evaluated point? Or generate one.
    
    # Let's generate a second point nearby
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
