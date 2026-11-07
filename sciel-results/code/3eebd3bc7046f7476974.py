import math
import time


def find_root(f, x0):
    deadline = time.time() + 140.0
    call_count = [0]

    def fc(x):
        call_count[0] += 1
        if call_count[0] >= 498:
            raise RuntimeError("call limit")
        return f(x)

    x = float(x0)
    fx = fc(x)
    if abs(fx) <= 1e-8:
        return x

    # Try secant with several small step sizes to find a good pair
    # For oscillatory functions, a small step often works well
    base_step = max(1e-4, 0.01 * max(1.0, abs(x)))
    
    best_x, best_fx = x, fx
    best_abs = abs(fx)

    # Try multiple step sizes for secant initialization
    step_multipliers = [1.0, 0.5, 2.0, 0.1, 5.0, 0.2, 10.0]
    
    secant_success = False
    for mult in step_multipliers:
        if time.time() > deadline or call_count[0] >= 490:
            break
        step = base_step * mult
        x1 = x + step
        fx1 = fc(x1)
        
        if abs(fx1) <= 1e-8:
            return x1
        if abs(fx1) < best_abs:
            best_x, best_fx, best_abs = x1, fx1, abs(fx1)
        
        # Check if we have a good secant pair (sign change or close values)
        if fx * fx1 < 0:
            # Sign change - great for secant
            break
        # Also try if values are small
        if abs(fx) < 1e-4 and abs(fx1) < 1e-4:
            break
    
    # Now run secant method
    x0_s, fx0_s = x, fx
    x1_s, fx1_s = x1, fx1
    
    for i in range(40):
        if time.time() > deadline or call_count[0] >= 495:
            break
        if abs(fx1_s) <= 1e-8:
            return x1_s
        if abs(fx0_s) <= 1e-8:
            return x0_s
        if fx0_s == fx1_s:
            # Degenerate - try a new step
            x2 = x0_s + 0.5 * (x1_s - x0_s)
            fx2 = fc(x2)
            if abs(fx2) <= 1e-8:
                return x2
            x0_s, fx0_s = x1_s, fx1_s
            x1_s, fx1_s = x2, fx2
            continue
        
        denom = fx1_s - fx0_s
        if abs(denom) < 1e-30:
            x2 = 0.5 * (x0_s + x1_s)
        else:
            x2 = x1_s - fx1_s * (x1_s - x0_s) / denom
        
        # Sanity check
        if not math.isfinite(x2) or abs(x2) > 1e15:
            x2 = 0.5 * (x0_s + x1_s)
        
        fx2 = fc(x2)
        
        if abs(fx2) <= 1e-8:
            return x2
        if abs(fx2) < best_abs:
            best_x, best_fx, best_abs = x2, fx2, abs(fx2)
        
        x0_s, fx0_s = x1_s, fx1_s
        x1_s, fx1_s = x2, fx2
    
    if abs(fx1_s) <= 1e-8:
        return x1_s
    if abs(fx0_s) <= 1e-8:
        return x0_s
    
    # If secant didn't converge well, try to find a bracket and use bisection
    # Search for sign change
    a, b, fa, fb = None, None, None, None
    step = max(1e-2, 0.1 * max(1.0, abs(x0)))
    
    for i in range(25):
        if time.time() > deadline or call_count[0] >= 495:
            break
        xa = x0 - step * (2 ** i)
        fa = fc(xa)
        if abs(fa) <= 1e-8:
            return xa
        if fx * fa < 0:
            a, b = xa, x
            fa, fb = fa, fx
            break
        
        xb = x0 + step * (2 ** i)
        fb = fc(xb)
        if abs(fb) <= 1e-8:
            return xb
        if fx * fb < 0:
            a, b = x, xb
            fa, fb = fx, fb
            break
        if fa * fb < 0:
            a, b = xa, xb
            fa, fb = fa, fb
            break
        
        if step * (2 ** i) > 1e10:
            break
    
    if a is not None:
        # Bisection (guaranteed convergence)
        for i in range(60):
            if time.time() > deadline or call_count[0] >= 498:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            m = 0.5 * (a + b)
            fm = fc(m)
            if abs(fm) <= 1e-8:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return a if abs(fa) <= abs(fb) else b
    
    # Return best found
    return best_x
