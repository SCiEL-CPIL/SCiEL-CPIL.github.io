"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 135  # safety margin before 150s kill
    max_evals = 500
    eval_count = [0]
    
    def safe_f(x):
        if time.time() > deadline:
            raise TimeoutError()
        if eval_count[0] >= max_evals:
            raise ValueError("max evals exceeded")
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            return val
        return float(val)
    
    x = float(x0)
    
    # Try to get a good initial bracket
    # First, evaluate at x0
    try:
        fx = safe_f(x)
    except (TimeoutError, ValueError):
        return x
    
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket around x0
    h = 1e-3 * max(1.0, abs(x))
    a, b = x - h, x + h
    fa, fb = None, None
    
    # Evaluate at a and b if within budget
    try:
        fa = safe_f(a)
    except (TimeoutError, ValueError):
        pass
    try:
        fb = safe_f(b)
    except (TimeoutError, ValueError):
        pass
    
    # If we have a bracket, use it
    if fa is not None and fb is not None and math.isfinite(fa) and math.isfinite(fb):
        if fa * fb < 0:
            # We have a sign change, use Brent's method
            return _brent(f, a, b, fa, fb, safe_f, deadline)
        # No sign change, try expanding
        for i in range(10):
            h *= 2
            a2, b2 = x - h, x + h
            try:
                fa2 = safe_f(a2)
                fb2 = safe_f(b2)
            except (TimeoutError, ValueError):
                break
            if not (math.isfinite(fa2) and math.isfinite(fb2)):
                break
            if fa2 * fb2 < 0:
                return _brent(f, a2, b2, fa2, fb2, safe_f, deadline)
            if abs(fa2) < abs(fa):
                a, fa = a2, fa2
            if abs(fb2) < abs(fb):
                b, fb = b2, fb2
            if time.time() > deadline:
                break
    else:
        # Couldn't get a bracket, fall back to Newton with finite differences
        pass
    
    # Fallback: Newton's method with finite difference derivative
    x_best = x
    fx_best = fx
    if abs(fx) <= 1e-8:
        return x
    
    for _ in range(50):
        if time.time() > deadline:
            break
        try:
            h = 1e-6 * max(1.0, abs(x))
            fx_h = safe_f(x + h)
            d = (fx_h - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break
            x = x_new
            fx = fx_new
            if abs(fx) < abs(fx_best):
                x_best = x
                fx_best = fx
            if abs(fx) <= 1e-8:
                return x
        except (TimeoutError, ValueError):
            break
    
    # Return the best x we found
    return x_best


def _brent(f, a, b, fa, fb, safe_f, deadline):
    """Brent's method implementation."""
    max_iter = 100
    tol = 1e-8
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    d = b - a
    e = d
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if abs(fb) <= tol:
            return b
        if abs(fc) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            b, c = c, a
            fb, fc = fc, fa
        
        m = 0.5 * (a - b)
        eps = 4 * 2.220446049250313e-16 * abs(b)
        cond1 = abs(m) <= eps
        cond2 = fa == fb
        
        if cond1 or cond2:
            # Bisection
            if a <= b:
                d = m
            else:
                d = -m
            e = d
        else:
            # Inverse quadratic interpolation
            S = fb / fa
            if a == c:
                P = 2 * m * S
                Q = 1 - S
            else:
                R = fa / fc
                S1 = fb / fc
                Q = R * (1 - S) * (1 - S1)
                P = (2 * m * Q - b - a * (S1 - 1) * (S - 1) * (S1 - 1)) 
                # Actually let me use the standard formula
                # P = S * (2*m*(fb/fa - 1) - (b-a)*(fb/fc - 1))
                # Let me redo this properly
                P = S * (2 * m * (S - 1) - (b - a) * (fb / fc - 1))
                Q = R * (1 - S) * (1 - S1)
            
            if P > 0:
                Q = -Q
            elif Q < 0:
                P = -P
            
            if 2 * P < min(3 * m * Q - abs(eps * Q), e * Q):
                e = d
                d = P / Q
            else:
                d = m
                e = d
        
        if a <= b:
            if d < 0:
                d = -d
        else:
            d = -d
        
        x = b + d
        
        # Ensure x is not too close to a or b
        if (x - a) < 0.5 * (b - a) or (b - x) < 0.5 * (b - a):
            x = 0.5 * (a + b)
        
        fx = safe_f(x)
        d = b - x
        e = d
        
        if abs(fx) < abs(fb):
            a, b = b, x
            fa, fb = fb, fx
        elif fx * fa > 0:
            a, c = x, a
            fa, fc = fx, fa
        else:
            b, c = x, b
            fb, fc = fx, fc
    
    # Return the best point
    if abs(fb) < abs(fa) and abs(fb) < abs(fc):
        return b
    elif abs(fa) < abs(fb) and abs(fa) < abs(fc):
        return a
    else:
        return c

# EVOLVE-BLOCK-END
