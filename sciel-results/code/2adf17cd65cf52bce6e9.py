"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    """
    Find a root of f near x0 using a dense grid scan to locate a sign-change
    bracket, followed by high-precision bisection/secant refinement.
    All function evaluations are cached to minimize total call count.
    """
    x0 = float(x0)
    cache = {}
    call_count = 0
    max_calls = 480  # stay well under 500

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

    # Step 1: Dense uniform grid scan around x0 to find a sign change
    # Try multiple scales to handle different magnitudes of roots
    base_scale = max(1.0, abs(x0))
    
    # First, try a fine local grid
    scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    n_points = 200  # points per half-range, so 400 total
    
    bracket = None
    for scale in scales:
        half_range = scale * base_scale
        if half_range < 1e-12:
            half_range = 1e-12
        # Create uniform grid from x0 - half_range to x0 + half_range
        xs = np.linspace(x0 - half_range, x0 + half_range, n_points)
        
        # Evaluate all points, respecting call budget
        fs = []
        valid_mask = []
        for x in xs:
            x = float(x)
            if x in cache:
                fs.append(cache[x])
                valid_mask.append(True)
            else:
                if call_count >= max_calls - 10:
                    fs.append(np.nan)
                    valid_mask.append(False)
                else:
                    val = eval_f(x)
                    fs.append(val)
                    valid_mask.append(np.isfinite(val))
        
        fs = np.array(fs)
        valid_mask = np.array(valid_mask)
        
        # Find sign changes among valid entries
        for i in range(len(xs) - 1):
            if not (valid_mask[i] and valid_mask[i+1]):
                continue
            f1 = fs[i]
            f2 = fs[i+1]
            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue
            if f1 * f2 <= 0:
                # Sign change found
                if abs(f1) <= 1e-8:
                    bracket = (float(xs[i]), float(xs[i]))
                    break
                if abs(f2) <= 1e-8:
                    bracket = (float(xs[i+1]), float(xs[i+1]))
                    break
                bracket = (float(xs[i]), float(xs[i+1]))
                break
            if abs(f1) <= 1e-8:
                bracket = (float(xs[i]), float(xs[i]))
                break
            if abs(f2) <= 1e-8:
                bracket = (float(xs[i+1]), float(xs[i+1]))
                break
        if bracket is not None:
            break
    
    # If no bracket found, try a coarser but wider scan
    if bracket is None:
        # Wider scan
        wide_scale = 1000.0 * base_scale
        n_wide = 100
        xs_wide = np.linspace(x0 - wide_scale, x0 + wide_scale, n_wide)
        for i in range(len(xs_wide) - 1):
            x1 = float(xs_wide[i])
            x2 = float(xs_wide[i+1])
            if call_count >= max_calls - 5:
                break
            f1 = eval_f(x1)
            f2 = eval_f(x2)
            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue
            if f1 * f2 <= 0:
                if abs(f1) <= 1e-8:
                    bracket = (x1, x1)
                    break
                if abs(f2) <= 1e-8:
                    bracket = (x2, x2)
                    break
                bracket = (x1, x2)
                break
            if abs(f1) <= 1e-8:
                bracket = (x1, x1)
                break
            if abs(f2) <= 1e-8:
                bracket = (x2, x2)
                break
        if bracket is None:
            # Last resort: return x0 if it's close enough, otherwise try x0 itself
            f0 = eval_f(x0)
            if np.isfinite(f0) and abs(f0) <= 1e-8:
                return x0
            # Try a few more points
            for dx in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                for sign in [1, -1]:
                    x_try = x0 + sign * dx * base_scale
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
    
    # Ensure fa and fb have opposite signs (or one is zero)
    if fa * fb > 0:
        # Shouldn't happen if bracket was properly identified, but handle it
        # Use secant method from a and b
        x = x0
        fx = eval_f(x)
        for _ in range(50):
            if abs(fx) <= 1e-10:
                return x
            if call_count >= max_calls:
                break
            h = 1e-8 * max(1.0, abs(x))
            fx2 = eval_f(x + h)
            d = (fx2 - fx) / h
            if d == 0 or not np.isfinite(d):
                break
            x_new = x - fx / d
            if not np.isfinite(x_new):
                break
            x = x_new
            fx = eval_f(x)
        return x
    
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
