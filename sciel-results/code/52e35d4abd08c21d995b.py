"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import brentq, newton, minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Uses a multi-strategy approach:
    1. Try Newton with finite differences first (fast if derivative is good)
    2. Try Brent's method in a local bracket
    3. Try a local minimization of f^2 (derivative-free)
    4. Multi-start random restarts with Brent
    """
    x0 = float(x0)
    best_x = x0
    best_f = abs(f(x0))
    
    # Track evaluations
    eval_count = 1  # already called f(x0)
    
    def try_newton(x_start, max_iter=50):
        nonlocal eval_count
        x = float(x_start)
        for _ in range(max_iter):
            if eval_count >= 490:
                return None
            fx = f(x)
            eval_count += 1
            if abs(fx) <= 1e-12:
                return x
            h = 1e-6 * max(1.0, abs(x))
            fxp = f(x + h)
            eval_count += 1
            d = (fxp - fx) / h
            if d == 0:
                return None
            x_new = x - fx / d
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + np.sign(x_new - x) * 10 * max(1.0, abs(x))
            x = x_new
            if not np.isfinite(x):
                return None
        return x if abs(f(x)) <= 1e-8 else None
    
    def try_brent(a, b, max_iter=100):
        nonlocal eval_count
        if eval_count >= 480:
            return None
        try:
            fa = f(a)
            fb = f(b)
            eval_count += 2
            if fa * fb > 0:
                return None
            # Use scipy brentq
            result = brentq(f, a, b, maxiter=max_iter, xtol=1e-12, rtol=1e-12)
            eval_count += 10  # rough estimate
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
        return None
    
    def try_local_minimize(x_start, max_iter=100):
        nonlocal eval_count
        if eval_count >= 480:
            return None
        try:
            # Minimize f(x)^2 locally
            res = minimize_scalar(lambda x: f(x)**2, bounds=(x_start - 1.0, x_start + 1.0), 
                                   method='bounded', options={'maxiter': max_iter, 'xatol': 1e-12})
            eval_count += res.nfev
            if res.fun <= 1e-16:
                return res.x
            if abs(f(res.x)) <= 1e-8:
                return res.x
        except Exception:
            pass
        return None
    
    # Strategy 1: Newton from x0
    x_newton = try_newton(x0)
    if x_newton is not None and abs(f(x_newton)) < best_f:
        best_x = x_newton
        best_f = abs(f(x_newton))
        if best_f <= 1e-8:
            return best_x
    
    # Strategy 2: Try to find a bracket around x0
    for scale in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        if eval_count >= 450:
            break
        a = x0 - scale
        b = x0 + scale
        x_bracket = try_brent(a, b)
        if x_bracket is not None:
            if abs(f(x_bracket)) < best_f:
                best_x = x_bracket
                best_f = abs(f(x_bracket))
            if best_f <= 1e-8:
                return best_x
    
    # Strategy 3: Local minimization of f^2
    x_min = try_local_minimize(x0)
    if x_min is not None:
        if abs(f(x_min)) < best_f:
            best_x = x_min
            best_f = abs(f(x_min))
        if best_f <= 1e-8:
            return best_x
    
    # Strategy 4: Multi-start random restarts
    np.random.seed(42)
    for i in range(20):
        if eval_count >= 450:
            break
        # Random restart around x0 with increasing radii
        radius = 0.1 * (i + 1) ** 1.5
        x_rand = x0 + np.random.uniform(-radius, radius)
        
        # Try Newton from random point
        x_n = try_newton(x_rand, max_iter=30)
        if x_n is not None:
            if abs(f(x_n)) < best_f:
                best_x = x_n
                best_f = abs(f(x_n))
            if best_f <= 1e-8:
                return best_x
        
        # Try Brent in a small bracket around random point
        for scale in [0.01, 0.1, 1.0]:
            if eval_count >= 450:
                break
            a = x_rand - scale
            b = x_rand + scale
            x_b = try_brent(a, b)
            if x_b is not None:
                if abs(f(x_b)) < best_f:
                    best_x = x_b
                    best_f = abs(f(x_b))
                if best_f <= 1e-8:
                    return best_x
    
    return best_x
# EVOLVE-BLOCK-END
