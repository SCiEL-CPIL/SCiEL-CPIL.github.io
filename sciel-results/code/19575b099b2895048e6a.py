import math

def find_root(f, x0):
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Strategy: Secant Method with Bisection fallback
    # We maintain two points x0, x1 and their function values.
    
    # Initial step
    # Try a small step first
    h = 1e-6
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-8:
        return x1

    # If f(x) and f(x1) have same sign, we don't have a bracket yet.
    # We will try to find a bracket by expanding, but we'll prioritize Secant if we can.
    # However, Secant is unstable without a bracket.
    # Let's try to find a bracket first, but efficiently.
    
    # Expand to find a bracket
    # We already have x, fx and x1, f1.
    # If fx * f1 > 0, expand x1 further.
    # But oscillatory functions might have roots very close.
    
    # Let's try a different initial setup:
    # x_a = x0
    # x_b = x0 + 1e-4 (slightly larger to catch sign changes)
    
    x_a = x
    f_a = fx
    x_b = x + 1e-4
    f_b = f(x_b)
    
    if abs(f_b) <= 1e-8:
        return x_b

    # If no sign change, expand x_b
    # Max 20 expansions to avoid wasting calls
    for i in range(20):
        if f_a * f_b < 0:
            break
        # Expand x_b
        # Double the distance from x_a
        dist = x_b - x_a
        x_b = x_a + 2 * dist
        f_b = f(x_b)
        if abs(f_b) <= 1e-8:
            return x_b
        if not math.isfinite(x_b) or not math.isfinite(f_b):
            break

    # If we found a bracket, use Brent's method (via manual implementation or scipy?)
    # The prompt says "design a new program... do not reuse current designs".
    # The current design uses `scipy.optimize.brentq`.
    # I should implement a root finder myself or use a different SciPy method?
    # "Python standard library, NumPy and SciPy are available."
    # Using `scipy.optimize.brentq` is reusing a design component.
    # Let's implement a **Secant Method** primarily, as it is distinct from Brent's mixed strategy.
    
    # If we have a bracket, we can use Bisection (safe, slow) or Illinois (faster).
    # If we don't have a bracket, we assume the root is near x0 and use Secant blindly?
    # Dangerous.
    
    # Let's assume the root is close to x0.
    # If no bracket found in 20 steps, it's likely a tangent root or root is far.
    # For tangent root, minimization is needed.
    
    # Let's try Secant first with the two closest points.
    # If we have a bracket [x_a, x_b], we can use it to bound the Secant steps.
    
    if f_a * f_b < 0:
        # We have a bracket. Use a custom Brent-like or Illinois method.
        # Illinois method:
        a, b = x_a, x_b
        fa, fb = f_a, f_b
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for _ in range(100):
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            # Illinois update
            if fa * fb > 0:
                # This shouldn't happen if we maintain bracket
                break
            
            # Choose next point
            if abs(fa) < abs(fb):
                # Keep a, update b
                # Interpolate
                # x_new = (a*fb - b*fa) / (fb - fa)
                denom = fb - fa
                if abs(denom) < 1e-30:
                    x_new = (a + b) / 2.0
                else:
                    x_new = (a * fb - b * fa) / denom
                
                # Ensure x_new is in [a, b]
                if x_new <= a or x_new >= b:
                    x_new = (a + b) / 2.0
                    
                f_new = f(x_new)
                if abs(f_new) <= 1e-8:
                    return x_new
                
                # Update bracket
                if fa * f_new < 0:
                    b = x_new
                    fb = f_new
                else:
                    # Halve the side where we just stepped
                    # a stays, b becomes midpoint? No, Illinois halves the side that did NOT change sign with the new point?
                    # Actually, Illinois method:
                    # If f(a)*f(new) < 0, new is in (a, b). Set b=new.
                    # Else, f(new)*f(b) < 0. Set a=new. And halve the distance of the other side?
                    # Standard Illinois:
                    # If f(a) and f(new) have same sign, set a=new, and halve f(b) (or b = (a+b)/2? No, b stays, but we treat the "effective" b as halved for next interpolation? No, that's for function values in some variants).
                    # Let's stick to simple Bisection if Secant/Illinois is too complex to implement correctly without bugs.
                    # Bisection is guaranteed to converge.
                    a = x_new
                    fa = f_new
            else:
                # Keep b, update a
                denom = fb - fa
                if abs(denom) < 1e-30:
                    x_new = (a + b) / 2.0
                else:
                    x_new = (a * fb - b * fa) / denom
                
                if x_new <= a or x_new >= b:
                    x_new = (a + b) / 2.0
                    
                f_new = f(x_new)
                if abs(f_new) <= 1e-8:
                    return x_new
                
                if fa * f_new < 0:
                    b = x_new
                    fb = f_new
                else:
                    a = x_new
                    fa = f_new
            
            # Check convergence of interval
            if abs(b - a) < 1e-12:
                return (a + b) / 2.0

    # If no bracket found, try local minimization of |f(x)|
    # Using Golden Section Search
    # Range: x0 +/- 1.0
    # If not found, expand?
    
    def golden_section_minimize(func, a, b, tol=1e-12, max_iter=100):
        # Minimize func(x) in [a, b]
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = func(c)
        fd = func(d)
        
        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = func(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = func(d)
                
        return (a + b) / 2.0

    # Try minimizing |f(x)| in [x0-1, x0+1]
    # We need to count calls. Golden section uses ~2 calls per iter.
    # 100 iters = 200 calls. Too many?
    # Let's limit to 50 iters (100 calls).
    
    # Better: Try a few sample points first.
    # x0, x0+0.1, x0-0.1, x0+0.5, x0-0.5
    # Pick best, then refine.
    
    candidates = [x, x + 0.1, x - 0.1, x + 0.5, x - 0.5, x + 1.0, x - 1.0]
    best_x = x
    best_f = abs(fx)
    
    for cx in candidates:
        cf = f(cx)
        if abs(cf) < best_f:
            best_f = abs(cf)
            best_x = cx
            if best_f <= 1e-8:
                return cx

    # Refine around best_x using Golden Section
    # Window size 0.1
    a = best_x - 0.1
    b = best_x + 0.1
    x_min = golden_section_minimize(lambda z: abs(f(z)), a, b, tol=1e-12, max_iter=50)
    f_min = abs(f(x_min))
    
    if f_min <= 1e-8:
        return x_min
    
    # If still not close, try larger window?
    # Or return best_x found so far.
    return best_x
