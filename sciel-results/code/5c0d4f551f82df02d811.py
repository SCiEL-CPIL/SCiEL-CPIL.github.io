import numpy as np
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = 100.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # 1. Try single-shot polynomial reconstruction (Degree 4)
    # Evaluate at 5 distinct points
    points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = []
    for p in points:
        if time.time() > deadline:
            break
        values.append(f(p))
        
    if len(values) == 5:
        try:
            # Fit a polynomial of degree 4
            # coeffs[0] is x^4, coeffs[1] is x^3, etc.
            coeffs = np.polyfit(points, values, 4)
            
            # Check if the leading coefficient is negligible (lower degree)
            # numpy.roots handles this, but we can be explicit if needed.
            roots = np.roots(coeffs)
            
            # Filter real roots
            real_roots = []
            for r in roots:
                if abs(r.imag) < 1e-9:
                    real_roots.append(r.real)
            
            if real_roots:
                # Find root closest to x0
                best_root = min(real_roots, key=lambda x: abs(x - x0))
                
                # Verify the root
                fx_best = f(best_root)
                if abs(fx_best) <= 1e-8:
                    return best_root
        except Exception:
            pass

    # 2. Fallback: Newton's method with finite difference derivative
    # Start from x0
    x = x0
    best_x = x
    best_f = f(x)
    
    if abs(best_f) <= 1e-8:
        return x

    for i in range(40):
        if time.time() > deadline:
            break
            
        fx = f(x)
        if abs(fx) < abs(best_f):
            best_x = x
            best_f = fx
            
        if abs(fx) <= 1e-8:
            return x
            
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        
        if d == 0:
            # Try a different step or break
            break
            
        new_x = x - fx / d
        
        # Damping if step is too large
        if abs(new_x - x) > 10 * max(1.0, abs(x)):
            new_x = x + 0.5 * (new_x - x)
            
        x = new_x
        
    return best_x
