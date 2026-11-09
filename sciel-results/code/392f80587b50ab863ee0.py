"""Root finding for ill-conditioned equations: aggressive hybrid with caching."""
import time
import math

def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin below 120s limit
    MAX_EVALS = 495   # Safety margin below 500 limit

    eval_count = 0
    cache = {}
    best_x = float(x0)
    best_f = None
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
        # Use a rounded key for caching to handle near-duplicates
        key = round(float(x), 14)
        if key in cache:
            val = cache[key]
            if math.isfinite(val):
                abs_val = abs(val)
                if abs_val < best_abs:
                    best_f = val
                    best_x = float(key)
                    best_abs = abs_val
                return val
        try:
            val = f(float(x))
            if not math.isfinite(val):
                cache[key] = val
                return None
            eval_count += 1
            cache[key] = val
            abs_val = abs(val)
            if abs_val < best_abs:
                best_f = val
                best_x = float(key)
                best_abs = abs_val
            return val
        except Exception:
            return None

    # Initial evaluation
    try:
        best_f = f(best_x)
        if math.isfinite(best_f):
            eval_count += 1
            best_abs = abs(best_f)
            cache[round(float(best_x), 14)] = best_f
    except Exception:
        best_f = float('inf')
        best_abs = float('inf')

    if best_f is not None and best_abs <= 1e-8:
        return best_x

    def polish(x_start, max_iter=100):
        """High-precision secant/Newton refinement."""
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        
        x1 = float(x_start)
        f1 = safe_eval(x1)
        if f1 is None:
            return None
        
        if abs(f1) <= 1e-8:
            return x1
        
        # Establish second point
        h = 1e-8 * max(1.0, abs(x1))
        x0_ = x1 + h
        f0_ = safe_eval(x0_)
        if f0_ is None:
            return None
            
        if abs(f0_) < abs(f1):
            x1, f1 = x0_, f0_
            x0_ = x1 - h
            f0_ = safe_eval(x0_)
            if f0_ is None:
                return None
            if abs(f0_) < abs(f1):
                x1, f1 = x0_, f0_

        for _ in range(max_iter):
            if check_budget():
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_
            
            # Secant step
            if f1 == f0_:
                x_new = x1 + 1e-10 * max(1.0, abs(x1))
            else:
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
            
            # Safeguard against huge jumps
            max_jump = max(1e-6, 0.5 * abs(x1))
            if abs(x_new - x1) > max_jump:
                x_new = x1 + math.copysign(max_jump, x_new - x1)
            
            f_new = safe_eval(x_new)
            if f_new is None:
                break
            
            x0_, f0_ = x1, f1
            x1, f1 = x_new, f_new
            
            if abs(x1 - x0_) < 1e-14 * max(1.0, abs(x1)):
                break
        
        if abs(f1) <= 1e-8:
            return x1
        return None

    def solve_local(x_start, max_steps=40):
        """Hybrid Bisection/Secant solver."""
        nonlocal eval_count, best_x, best_f, best_abs
        
        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
            
        if abs(fx) <= 1e-8:
            return x

        # Establish bracket or secant pair
        h = 1e-4 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        
        if fx2 is None:
            return x
            
        if fx * fx2 > 0:
            # No bracket, try expanding or just use secant
            # Try to find a bracket by expanding
            bracketed = False
            for i in range(10):
                h = h * 2
                x2 = x + h
                fx2 = safe_eval(x2)
                if fx2 is None:
                    break
                if fx * fx2 < 0:
                    bracketed = True
                    break
            if not bracketed:
                # Fallback to secant-like behavior without strict bracket
                # Just run secant iterations
                x1, f1 = x, fx
                x0_, f0_ = x2, fx2
                for _ in range(max_steps):
                    if check_budget():
                        break
                    if abs(f1) <= 1e-8:
                        return x1
                    if abs(f0_) <= 1e-8:
                        return x0_
                    if f1 == f0_:
                        break
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
                return x1 if abs(f1) < best_abs else x
        else:
            # We have a bracket [x, x2]
            a, fa = x, fx
            b, fb = x2, fx2
            for _ in range(max_steps):
                if check_budget():
                    break
                if abs(fa) <= 1e-8:
                    return a
                if abs(fb) <= 1e-8:
                    return b
                
                # Secant step
                if fa != fb:
                    x_new = a - fa * (a - b) / (fa - fb)
                else:
                    x_new = (a + b) / 2.0
                
                # If secant step is outside bracket, use bisection
                if x_new <= a or x_new >= b:
                    x_new = (a + b) / 2.0
                
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                    
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
                    
                if abs(b - a) < 1e-12 * max(1.0, abs(a)):
                    break
            
            # Return the better of the bracket endpoints
            if abs(fa) < abs(fb):
                return a
            else:
                return b

    # Main Strategy 1: Local solve from x0
    result = solve_local(x0, max_steps=50)
    if best_abs <= 1e-8:
        return best_x

    # Main Strategy 2: Perturbed starts
    perturbations = [
        1e-2, -1e-2,
        1e-3, -1e-3,
        1e-1, -1e-1,
        0.5, -0.5,
        1.0, -1.0,
        10.0, -10.0
    ]
    
    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        result = solve_local(x_pert, max_steps=30)
        if best_abs <= 1e-8:
            return best_x

    # Main Strategy 3: Broader sampling
    radii = [10.0, 100.0, 1000.0, 10000.0]
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
                result = solve_local(x_s, max_steps=20)
                if best_abs <= 1e-8:
                    return best_x

    # Main Strategy 4: Final polish attempt on best point
    if best_abs < 1e-4:
        result = polish(best_x, max_iter=100)
        if result is not None:
            return best_x

    return best_x
