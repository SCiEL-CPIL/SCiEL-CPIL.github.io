"""Brent's method with adaptive bracketing for oscillatory root finding."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 140.0          # safety margin before 150 s hard kill
    TOL = 1e-12               # aim well below 1e-8
    HARD_TOL = 1e-8           # contract tolerance
    MAX_CALLS = 450           # Hard limit on calls to ensure safety

    call_count = 0

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    # Evaluate start
    f0 = safe_f(x0)
    best_x = x0
    best_val = abs(f0) if f0 != float('nan') else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx != float('nan'):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with a sign change
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    eval_cache = {x0: f0}
    
    def get_f(x):
        if x in eval_cache:
            return eval_cache[x]
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    # Strategy: Scan outward from x0 with increasing step sizes.
    scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    bracket_found = False
    
    for scale in scales:
        if not elapsed_ok() or call_count >= MAX_CALLS - 20:
            break
            
        step = scale / 10.0
        prev_x = None
        prev_f = None
        
        for k in range(-10, 11):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
                
            x_test = x0 + k * step
            f_test = get_f(x_test)
            
            if f_test != float('nan'):
                if abs(f_test) <= HARD_TOL:
                    return x_test
                    
                if prev_x is not None and prev_f is not None:
                    if prev_f * f_test < 0:
                        a, b = prev_x, x_test
                        fa, fb = prev_f, f_test
                        bracket_found = True
                        break
                prev_x = x_test
                prev_f = f_test
            
            if bracket_found:
                break
        
        if bracket_found:
            break

    # If still no bracket, try geometric expansion in both directions
    if not bracket_found:
        h = max(1e-4, abs(x0) * 1e-4)
        last_pos_x = x0
        last_pos_f = f0
        last_neg_x = x0
        last_neg_f = f0
        
        for i in range(1, 30):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
                
            # Positive side
            x_new_pos = x0 + i * h
            f_new_pos = get_f(x_new_pos)
            
            if f_new_pos != float('nan') and last_pos_f != float('nan'):
                if f_new_pos * last_pos_f < 0:
                    a, b = last_pos_x, x_new_pos
                    fa, fb = last_pos_f, f_new_pos
                    bracket_found = True
                    break
                elif abs(f_new_pos) <= HARD_TOL:
                    return x_new_pos
            
            if f_new_pos != float('nan'):
                last_pos_x = x_new_pos
                last_pos_f = f_new_pos
                
            # Negative side
            x_new_neg = x0 - i * h
            f_new_neg = get_f(x_new_neg)
            
            if f_new_neg != float('nan') and last_neg_f != float('nan'):
                if f_new_neg * last_neg_f < 0:
                    a, b = x_new_neg, last_neg_x
                    fa, fb = f_new_neg, last_neg_f
                    bracket_found = True
                    break
                elif abs(f_new_neg) <= HARD_TOL:
                    return x_new_neg
            
            if f_new_neg != float('nan'):
                last_neg_x = x_new_neg
                last_neg_f = f_new_neg
                
            h *= 2.0

    # ------------------------------------------------------------------
    # 2) Refine with Brent's method
    # ------------------------------------------------------------------
    if a is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Brent's method implementation
        # c is the previous b, fc is the previous fb
        c = b
        fc = fb
        d = 0.0  # last step
        e = 0.0  # last step before d
        
        for _ in range(300):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break
                
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
            # Ensure |fa| <= |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa
                c, fc = a, fa
                d = b - a
                e = d
            
            # Stop if root is found
            if fa == fb:
                return a
            
            # Check for convergence
            tol1 = 2e-16 * max(abs(a), abs(b))
            tol2 = 0.5 * (b - a)
            
            if abs(b - a) < max(tol1, tol2):
                return a if abs(fa) < abs(fb) else b
            
            # Inverse quadratic interpolation
            s = None
            if c != a and c != b:
                # c, fc, a, fa, b, fb are all different
                t1 = (a - c) * (fa - fc) / (fc - fa)
                t2 = (a - b) * (fa - fb) / (fb - fa)
                t3 = (b - c) * (fb - fc) / (fc - fb)
                denom = t1 * t2 * t3
                if denom != 0:
                    s = a + (t1 * t2 * t3) / (t1 * t2 + t1 * t3 + t2 * t3)
            
            if s is None or not math.isfinite(s) or not (min(a, b) < s < max(a, b)):
                # Secant method
                if fa != fb:
                    s = (a * fb - b * fa) / (fb - fa)
                
                if s is None or not math.isfinite(s) or not (min(a, b) < s < max(a, b)):
                    # Bisection
                    s = 0.5 * (a + b)
            
            fs = get_f(s)
            if fs != float('nan'):
                if abs(fs) <= TOL:
                    return s
                
                # Update bracket
                if fa * fs < 0:
                    b, fb = s, fs
                    c, fc = b, fb
                    d = e
                    e = b - a
                elif fb * fs < 0:
                    c, fc = b, fb
                    b, fb = s, fs
                    d = e
                    e = b - a
                else:
                    c, fc = b, fb
                    d = e
                    e = 0.5 * e
                
                track(s, fs)
            
            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                return a if abs(fa) < abs(fb) else b

    # ------------------------------------------------------------------
    # 3) Final fallback
    # ------------------------------------------------------------------
    
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
