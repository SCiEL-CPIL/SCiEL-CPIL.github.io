import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s kill
    MAX_CALLS = 490  # stay under 500
    
    x = float(x0)
    calls = 0
    best_x = x
    best_fx = float('inf')
    
    def eval_f(x_val):
        nonlocal calls, best_x, best_fx
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() > deadline:
            return float('nan')
        calls += 1
        try:
            val = f(x_val)
        except:
            return float('nan')
        if not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_fx:
            best_fx = abs_val
            best_x = x_val
        return val

    # Initial evaluation
    fx0 = eval_f(x)
    if math.isnan(fx0):
        return best_x
    if abs(fx0) <= 1e-8:
        return x
    
    # Illinois method for refinement
    def illinois(a, fa, b, fb):
        nonlocal calls
        for _ in range(150):
            if calls >= MAX_CALLS or time.time() > deadline:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Illinois step: linear interpolation
            if fa == fb:
                c = (a + b) / 2.0
            else:
                c = (a * fb - b * fa) / (fb - fa)
            
            # Clamp to interval
            if not (a < c < b) or not math.isfinite(c):
                c = (a + b) / 2.0
                
            fc = eval_f(c)
            if math.isnan(fc):
                # Fallback to bisection
                c = (a + b) / 2.0
                fc = eval_f(c)
                if math.isnan(fc):
                    break
                    
            if abs(fc) <= 1e-8:
                return c
                
            if fa * fc < 0:
                b, fb = c, fc
            else:
                # Illinois: halve the side with smaller |f|
                if abs(fa) < abs(fb):
                    fa = fa / 2.0
                else:
                    fb = fb / 2.0
                a, fa = c, fc
                
            if abs(b - a) < 1e-13:
                break
                
        # Return better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # Phase 1: Find a bracket via expanding search
    # Try small steps first, then geometric expansion
    base_step = 0.1 * max(1.0, abs(x))
    
    # Fine local scan first (catches nearby roots)
    local_step = 0.01 * max(1.0, abs(x))
    prev_x = x
    prev_fx = fx0
    
    for i in range(1, 30):
        if calls >= MAX_CALLS or time.time() > deadline:
            break
        for direction in [1, -1]:
            if calls >= MAX_CALLS or time.time() > deadline:
                break
            test_x = x + direction * i * local_step
            test_fx = eval_f(test_x)
            if math.isnan(test_fx):
                break
            if abs(test_fx) <= 1e-8:
                return test_x
            if prev_fx * test_fx < 0:
                # Found bracket
                if prev_x < test_x:
                    result = illinois(prev_x, prev_fx, test_x, test_fx)
                else:
                    result = illinois(test_x, test_fx, prev_x, prev_fx)
                return result
            prev_x = test_x
            prev_fx = test_fx
    
    # Geometric expansion in both directions
    step = base_step
    L, R = x, x
    fL, fR = fx0, fx0
    
    for i in range(60):
        if calls >= MAX_CALLS or time.time() > deadline:
            break
            
        new_L = L - step
        new_R = R + step
        
        f_new_L = eval_f(new_L)
        f_new_R = eval_f(new_R)
        
        if not math.isnan(f_new_L):
            if abs(f_new_L) <= 1e-8:
                return new_L
            if f_new_L * fL < 0:
                result = illinois(new_L, f_new_L, L, fL)
                return result
            L, fL = new_L, f_new_L
        
        if not math.isnan(f_new_R):
            if abs(f_new_R) <= 1e-8:
                return new_R
            if f_new_R * fR < 0:
                result = illinois(R, fR, new_R, f_new_R)
                return result
            R, fR = new_R, f_new_R
        
        step *= 1.8
        if step > 1e8:
            break
    
    # Phase 2: If no bracket found, try a broader geometric scan
    # with different starting offsets
    for offset_mult in [0.05, 0.2, 0.5, 2.0, 5.0]:
        if calls >= MAX_CALLS or time.time() > deadline:
            break
        step = offset_mult * max(1.0, abs(x))
        prev_x = x
        prev_fx = fx0
        for i in range(1, 40):
            if calls >= MAX_CALLS or time.time() > deadline:
                break
            for direction in [1, -1]:
                if calls >= MAX_CALLS or time.time() > deadline:
                    break
                test_x = x + direction * step
                test_fx = eval_f(test_x)
                if math.isnan(test_fx):
                    break
                if abs(test_fx) <= 1e-8:
                    return test_x
                if prev_fx * test_fx < 0:
                    if prev_x < test_x:
                        result = illinois(prev_x, prev_fx, test_x, test_fx)
                    else:
                        result = illinois(test_x, test_fx, prev_x, prev_fx)
                    return result
                prev_x = test_x
                prev_fx = test_fx
            step *= 2.0
    
    # Return best x found
    return best_x

# EVOLVE-BLOCK-START
# (The above code is the implementation)
# EVOLVE-BLOCK-END
