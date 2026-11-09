# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    """
    Find a root of f near x0 using a geometric expansion of the search range
    with coarse grid scans to locate a sign-change bracket, followed by
    high-precision bisection/secant refinement.
    All function evaluations are cached to minimize total call count.
    """
    x0 = float(x0)
    cache = {}
    call_count = 0
    max_calls = 450  # Stay well under 500

    def eval_f(x):
        nonlocal call_count
        key = float(x)
        if key in cache:
            return cache[key]
        if call_count >= max_calls:
            raise RuntimeError("Too many calls")
        val = float(f(key))
        cache[key] = val
        call_count += 1
        return val

    # Step 1: Geometric expansion to find a bracket with a sign change
    # We expand the range around x0: [x0 - R, x0 + R]
    # Start with a small R and increase it.
    
    # Base scale for expansion
    base_scale = max(1e-3, abs(x0) * 1e-3, 1e-6)
    
    # Number of points in the coarse grid. 
    # Fewer points = fewer calls, but risk missing a sign change if oscillations are fast.
    # 50 points means 49 intervals.
    n_points = 50
    
    bracket = None
    current_radius = base_scale
    max_radius = 1e6 * max(1.0, abs(x0))
    
    # Try expanding the radius
    while current_radius <= max_radius:
        # Generate grid points
        xs = np.linspace(x0 - current_radius, x0 + current_radius, n_points)
        
        # Evaluate points
        fs = []
        valid = True
        for x in xs:
            x = float(x)
            if x in cache:
                fs.append(cache[x])
            else:
                if call_count >= max_calls - 10:
                    valid = False
                    break
                val = eval_f(x)
                fs.append(val)
        
        if not valid:
            break
            
        fs = np.array(fs)
        
        # Check for sign changes or zeros
        # We check pairs (i, i+1)
        for i in range(len(xs) - 1):
            f1 = fs[i]
            f2 = fs[i+1]
            
            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue
                
            # Check for exact zero or very small value
            if abs(f1) <= 1e-8:
                bracket = (float(xs[i]), float(xs[i]))
                break
            if abs(f2) <= 1e-8:
                bracket = (float(xs[i+1]), float(xs[i+1]))
                break
                
            # Check for sign change
            if f1 * f2 < 0:
                bracket = (float(xs[i]), float(xs[i+1]))
                break
        
        if bracket is not None:
            break
            
        # Expand radius
        current_radius *= 10.0
        
    # If no bracket found in geometric expansion, try a wider, sparser scan
    if bracket is None:
        # Fallback: Try a very wide range with fewer points
        wide_radius = 1e6 * max(1.0, abs(x0))
        n_wide = 20
        xs_wide = np.linspace(x0 - wide_radius, x0 + wide_radius, n_wide)
        for i in range(len(xs_wide) - 1):
            x1 = float(xs_wide[i])
            x2 = float(xs_wide[i+1])
            if call_count >= max_calls - 5:
                break
            f1 = eval_f(x1)
            f2 = eval_f(x2)
            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue
            if abs(f1) <= 1e-8:
                bracket = (x1, x1)
                break
            if abs(f2) <= 1e-8:
                bracket = (x2, x2)
                break
            if f1 * f2 < 0:
                bracket = (x1, x2)
                break
        
    # If still no bracket, return best effort (x0 or nearby)
    if bracket is None:
        f0 = eval_f(x0)
        if np.isfinite(f0) and abs(f0) <= 1e-8:
            return x0
        # Try a few local points
        for dx in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            for sign in [1, -1]:
                x_try = x0 + sign * dx * max(1.0, abs(x0))
                if call_count >= max_calls:
                    break
                f_try = eval_f(x_try)
                if np.isfinite(f_try) and abs(f_try) <= 1e-8:
                    return x_try
        return x0

    # Step 2: Refine the bracket using bisection + secant hybrid
    a, b = bracket
    fa = eval_f(a)
    fb = eval_f(b)
    
    # If already good enough
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b
    
    # Ensure fa and fb have opposite signs
    if fa * fb > 0:
        # This can happen if the bracket was formed by two points very close to zero 
        # but same sign, or if the function touches zero without crossing.
        # In that case, we just pick the one with smaller absolute value.
        # Or try a local search.
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # Bisection with secant acceleration
    x_best = a if abs(fa) < abs(fb) else b
    f_best = min(abs(fa), abs(fb))
    
    for _ in range(60):
        if call_count >= max_calls:
            break
        if abs(fa) <= 1e-10:
            return a
        if abs(fb) <= 1e-10:
            return b
        
        # Try secant step
        if fa != fb:
            x_sec = a - fa * (b - a) / (fb - fa)
            # Check if secant point is within bracket
            if a <= x_sec <= b:
                f_sec = eval_f(x_sec)
                if np.isfinite(f_sec):
                    if abs(f_sec) < f_best:
                        x_best = x_sec
                        f_best = abs(f_sec)
                    if abs(f_sec) <= 1e-10:
                        return x_sec
                    # Update bracket
                    if fa * f_sec <= 0:
                        b = x_sec
                        fb = f_sec
                    else:
                        a = x_sec
                        fa = f_sec
                    continue
        
        # Bisection
        x_mid = (a + b) / 2.0
        f_mid = eval_f(x_mid)
        if not np.isfinite(f_mid):
            break
        if abs(f_mid) < f_best:
            x_best = x_mid
            f_best = abs(f_mid)
        if abs(f_mid) <= 1e-10:
            return x_mid
        
        if fa * f_mid <= 0:
            b = x_mid
            fb = f_mid
        else:
            a = x_mid
            fa = f_mid
        
        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(x_mid)):
            break
    
    # Return the best point found
    if abs(f_best) <= 1e-8:
        return x_best
    # Final check on endpoints
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b
    return x_best

# EVOLVE-BLOCK-END
