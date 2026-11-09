"""Root finding for ill-conditioned equations: Global Optimization via Nelder-Mead on f(x)^2."""
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

    # 2. Nelder-Mead Simplex Method
    # We minimize g(x) = f(x)^2.
    # Since g(x) >= 0, minimizing it drives f(x) towards 0.
    # This is robust for ill-conditioned problems because it does not rely on 
    # local derivatives or sign changes, and can navigate "flat" or "steep" 
    # landscapes by expanding/contracting the simplex.
    
    # Initialize simplex
    # In 1D, the simplex has 2 vertices: x0 and x0 + delta.
    # Choose delta based on scale of x0.
    scale = max(1.0, abs(float(x0)))
    delta = 1e-3 * scale
    
    # Vertices: (x, g(x))
    # We store the function value of the surrogate g(x) = f(x)^2
    # But we only have f(x). We can compute g(x) = f(x)**2 on the fly.
    # However, if f(x) is very large, f(x)**2 might overflow. 
    # For root finding, we care about |f(x)|. 
    # Let's use |f(x)| as the objective for Nelder-Mead? 
    # No, |f(x)| is not smooth at 0. f(x)^2 is smooth.
    # If f(x) is huge, we can cap it or just use it. 
    # Let's use f(x)**2 but be careful with overflow.
    
    x_a = float(x0)
    x_b = x0 + delta
    
    g_a = None
    g_b = None
    
    fa = safe_eval(x_a)
    if fa is not None:
        g_a = fa * fa
    else:
        g_a = float('inf')
        
    fb = safe_eval(x_b)
    if fb is not None:
        g_b = fb * fb
    else:
        g_b = float('inf')

    # If both are inf, we might be in a bad spot. Try expanding?
    # For now, proceed. Nelder-Mead handles inf by expanding/reflecting.
    
    # Nelder-Mead parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink
    
    max_iter = 200
    tol = 1e-12
    
    for _ in range(max_iter):
        if check_budget():
            break
            
        # Sort vertices by objective value (g)
        # In 1D, we just have two points.
        # Let's maintain a list of points for generality, or just handle 2 points.
        # With 2 points, it's essentially a bracketing + interpolation method if we add a 3rd?
        # Standard Nelder-Mead in 1D uses 2 points? No, simplex in 1D has 2 vertices.
        # But standard NM uses n+1 vertices. So 2 vertices.
        # Operations:
        # 1. Sort: x_best, x_worst
        # 2. Reflect: x_r = x_best - alpha * (x_worst - x_best)
        # 3. If g(x_r) < g(x_best): Expand x_e = x_best + gamma * (x_r - x_best)
        #    If g(x_e) < g(x_r): x_new = x_e else x_new = x_r
        # 4. If g(x_r) >= g(x_best) and g(x_r) < g(x_worst): x_new = x_r
        # 5. If g(x_r) >= g(x_worst): Contract x_c = x_best + rho * (x_worst - x_best)
        #    If g(x_c) < g(x_worst): x_new = x_c
        #    Else Shrink: x_new = x_best + sigma * (x_worst - x_best)
        
        if g_a <= g_b:
            x_best, g_best = x_a, g_a
            x_worst, g_worst = x_b, g_b
        else:
            x_best, g_best = x_b, g_b
            x_worst, g_worst = x_a, g_a
            
        # Convergence check
        if abs(x_worst - x_best) < tol and abs(g_worst - g_best) < 1e-10:
            break
            
        # Reflection
        x_r = x_best - alpha * (x_worst - x_best)
        f_r = safe_eval(x_r)
        if f_r is None:
            break
        g_r = f_r * f_r
        
        if g_r < g_best:
            # Expansion
            x_e = x_best + gamma * (x_r - x_best)
            f_e = safe_eval(x_e)
            if f_e is None:
                break
            g_e = f_e * f_e
            
            if g_e < g_r:
                x_worst, g_worst = x_e, g_e
            else:
                x_worst, g_worst = x_r, g_r
        elif g_r < g_worst:
            # Reflection is acceptable
            x_worst, g_worst = x_r, g_r
        else:
            # Contraction
            x_c = x_best + rho * (x_worst - x_best)
            f_c = safe_eval(x_c)
            if f_c is None:
                break
            g_c = f_c * f_c
            
            if g_c < g_worst:
                x_worst, g_worst = x_c, g_c
            else:
                # Shrink
                # Move worst towards best
                x_worst = x_best + sigma * (x_worst - x_best)
                f_w = safe_eval(x_worst)
                if f_w is None:
                    break
                g_worst = f_w * f_w
        
        # Update vertices
        x_a = x_best
        x_b = x_worst
        g_a = g_best
        g_b = g_worst
        
        # Early exit if we found a root
        if g_best <= (1e-8)**2:
            return x_best

    # 3. Polish with Secant/Newton if we have a good bracket or close points
    # Nelder-Mead might stop near the root but not exactly at |f(x)| <= 1e-8.
    # We can try a few secant steps from the best point if we have another point nearby.
    
    # We have x_a and x_b. Let's try secant from the two best points.
    # Identify the two points with smallest |f|
    # We already track best_x. Let's find the second best from our history?
    # We didn't store history. Let's just use x_a and x_b.
    
    # Sort x_a, x_b by |f|
    fa = safe_eval(x_a) if g_a != float('inf') else None
    fb = safe_eval(x_b) if g_b != float('inf') else None
    
    # If we have two valid points, try secant
    if fa is not None and fb is not None and fa != fb:
        x1, f1 = x_a, fa
        x2, f2 = x_b, fb
        
        # Ensure f1 is smaller? Not strictly necessary for secant.
        for _ in range(50):
            if check_budget():
                break
            if f1 == f2:
                break
            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
            
            # Safeguard: don't jump too far
            max_jump = max(1.0, 0.5 * abs(x1))
            if abs(x_new - x1) > max_jump:
                x_new = x1 + math.copysign(max_jump, x_new - x1)
                
            f_new = safe_eval(x_new)
            if f_new is None:
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            # Update: keep the point with smaller |f|
            if abs(f_new) < abs(f1):
                x2, f2 = x1, f1
                x1, f1 = x_new, f_new
            else:
                x2, f2 = x_new, f_new
                
            if abs(x_new - x2) < 1e-14:
                break

    return best_x

# EVOLVE-BLOCK-END
