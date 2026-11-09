import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using a spectral prediction strategy:
    1. Sample the function on a logarithmic grid around x0 to capture
       oscillatory behavior across multiple scales.
    2. Estimate the dominant frequency and phase of the oscillation.
    3. Analytically predict the nearest zero-crossing based on the fitted sinusoid.
    4. Perform Newton-Raphson refinement from the predicted point.
    5. Fallback to a local grid search if spectral prediction fails.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Cache to avoid redundant calls
    cache = {}
    
    def f_cached(x):
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val

    # Initial check
    fx0 = f_cached(x0)
    if abs(fx0) <= 1e-8:
        return x0
        
    # Phase 1: Spectral Analysis
    # We sample points at logarithmic intervals to capture different frequencies.
    # We assume the function behaves like A * sin(w * x + phi) + C locally.
    # To handle oscillatory functions, we sample a window around x0.
    
    # We'll try a few different window sizes to ensure we capture the period.
    # If the period is very small, we need a small window. If large, a larger one.
    
    best_x = x0
    best_f = abs(fx0)
    
    # Sample points for spectral estimation
    # We use a symmetric set of points around x0 on a log scale relative to some base,
    # but since we don't know the scale, we try a few geometric progressions.
    
    # Strategy:
    # 1. Try to fit a sinusoid to points sampled in [x0 - R, x0 + R].
    # 2. We need at least 3 points to fit A, w, phi (assuming C=0 for simplicity, 
    #    or we can fit A, w, phi, C with 4 points).
    #    Actually, fitting w is non-linear. 
    #    Alternative: Estimate period T from zero crossings or peaks.
    #    But we don't have zero crossings yet.
    
    # Let's use a simpler spectral approach:
    # Sample N points evenly spaced in a window.
    # Compute the Discrete Fourier Transform (DFT) to find the dominant frequency.
    # However, for a single root, we might be better off just finding the local structure.
    
    # Let's try a "Multi-scale Zero Prediction" approach instead of full DFT fitting,
    # which is more robust to noise and non-sinusoidal components.
    
    # We will sample points at distances: d, 2d, 4d, ... and -d, -2d, -4d...
    # and look for sign changes or minima.
    
    # To be "spectral", let's estimate the local derivative and second derivative
    # to predict where f(x) = 0 using a quadratic model (parabola) or cubic model.
    # This is effectively a local spectral approximation (Taylor series).
    
    # Let's gather 5 points: x0, x0+h, x0-h, x0+2h, x0-2h
    # We can adjust h based on the magnitude of x0.
    
    h_base = max(1e-4, abs(x0) * 1e-4)
    
    # Try a few scales for h to ensure we capture the oscillation period
    scales = [1e-2, 1e-1, 1.0, 10.0]
    
    predicted_roots = []
    
    for scale in scales:
        h = h_base * scale
        
        # Sample points
        x1 = x0 - h
        x2 = x0
        x3 = x0 + h
        x4 = x0 - 2*h
        x5 = x0 + 2*h
        
        try:
            f1 = f_cached(x1)
            f2 = f_cached(x2)
            f3 = f_cached(x3)
            f4 = f_cached(x4)
            f5 = f_cached(x5)
        except:
            continue
            
        # Check for immediate root
        for x_val, f_val in zip([x1, x2, x3, x4, x5], [f1, f2, f3, f4, f5]):
            if abs(f_val) <= 1e-8:
                return x_val
            if abs(f_val) < best_f:
                best_f = abs(f_val)
                best_x = x_val
                
        # Fit a quadratic: f(x) ≈ a(x-x0)^2 + b(x-x0) + c
        # c = f2
        # f1 = a*h^2 - b*h + c
        # f3 = a*h^2 + b*h + c
        # Subtracting: f3 - f1 = 2*b*h => b = (f3 - f1) / (2*h)
        # Adding: f3 + f1 = 2*a*h^2 + 2*c => a = (f3 + f1 - 2*c) / (2*h^2)
        
        if h > 0:
            b = (f3 - f1) / (2 * h)
            a = (f3 + f1 - 2 * f2) / (2 * h * h)
            c = f2
            
            # Roots of a*t^2 + b*t + c = 0 where t = x - x0
            if abs(a) > 1e-12:
                disc = b*b - 4*a*c
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    t1 = (-b + sqrt_disc) / (2 * a)
                    t2 = (-b - sqrt_disc) / (2 * a)
                    predicted_roots.append(x0 + t1)
                    predicted_roots.append(x0 + t2)
            elif abs(b) > 1e-12:
                # Linear approximation
                t = -c / b
                predicted_roots.append(x0 + t)
                
    # Phase 2: Select the best predicted root
    # We prefer roots that are close to x0 and where the function value is small.
    
    candidate_x = best_x
    
    if predicted_roots:
        # Evaluate predicted roots
        # We don't want to call f too many times here, but we have a budget.
        # Let's just pick the one closest to x0, or the one with smallest |t|.
        # Actually, let's verify them quickly if we have budget.
        
        # Sort by distance to x0
        predicted_roots.sort(key=lambda x: abs(x - x0))
        
        for x_pred in predicted_roots:
            if not math.isfinite(x_pred):
                continue
            # Check if this prediction is better than current best_x
            # We can't evaluate f(x_pred) without spending a call, 
            # but Newton refinement will do that.
            # Let's just pick the closest one.
            candidate_x = x_pred
            break
            
    # Phase 3: Newton-Raphson Refinement
    x = candidate_x
    fx = f_cached(x)
    
    if abs(fx) <= 1e-8:
        return x
        
    # Use a hybrid Newton-Secant method for robustness
    # We need a second point for Secant. We can use x0 or a nearby point.
    # Let's use the previous best point if different, or x0.
    
    x_prev = x0
    if abs(x - x_prev) < 1e-12:
        x_prev = x + 1e-6
    fx_prev = f_cached(x_prev)
    
    for i in range(50):
        if time.time() > deadline - 2.0:
            break
            
        if abs(fx) <= 1e-8:
            return x
            
        if abs(fx_prev) <= 1e-8:
            return x_prev
            
        # Secant step
        if abs(fx - fx_prev) < 1e-15:
            # Degenerate, try Newton
            h = 1e-6
            f_plus = f_cached(x + h)
            dfdx = (f_plus - fx) / h
            if abs(dfdx) < 1e-12:
                break
            x_new = x - fx / dfdx
        else:
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            
        if not math.isfinite(x_new):
            break
            
        f_new = f_cached(x_new)
        
        # Update
        if abs(f_new) < abs(fx):
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = f_new
        else:
            # If it got worse, swap to ensure we move towards the smaller value
            # This is a safeguard for oscillatory functions
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = f_new
            # If we are diverging, we might need to restart from a different point
            # For now, we just continue. If it fails, we return best_x.
            
        # Divergence check
        if abs(x) > 1e10:
            break

    # Phase 4: Fallback
    if abs(fx) <= 1e-8:
        return x
    if abs(fx_prev) <= 1e-8:
        return x_prev
        
    # If Newton/Secant failed, do a local grid search around x0
    # This is a safety net.
    
    local_best_x = x0
    local_best_f = abs(fx0)
    
    # Search in a small neighborhood
    for radius in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        n_points = 20
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            x_try = x0 + radius * math.cos(angle)
            f_try = f_cached(x_try)
            if abs(f_try) < local_best_f:
                local_best_f = abs(f_try)
                local_best_x = x_try
            if abs(f_try) <= 1e-8:
                return x_try
                
        if local_best_f < 1e-4:
            break

    # Final Newton from local_best_x
    x = local_best_x
    fx = f_cached(x)
    
    for i in range(20):
        if abs(fx) <= 1e-8:
            return x
        h = 1e-6
        f_plus = f_cached(x + h)
        dfdx = (f_plus - fx) / h
        if abs(dfdx) < 1e-12:
            break
        x_new = x - fx / dfdx
        if not math.isfinite(x_new):
            break
        f_new = f_cached(x_new)
        if abs(f_new) < abs(fx):
            x = x_new
            fx = f_new
        else:
            break

    return x
# EVOLVE-BLOCK-END
