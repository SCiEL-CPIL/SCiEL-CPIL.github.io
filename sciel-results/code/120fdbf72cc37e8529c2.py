"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import brentq, newton, minimize_scalar

# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Uses an interleaved multi-start strategy with Safeguarded Inverse Quadratic Interpolation (SIQI).
    """
    x0 = float(x0)
    
    # State
    eval_count = 0
    max_evals = 500
    
    # Wrapped function to count evaluations
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        return f(x)

    best_x = x0
    best_f = abs(f_counted(x0))
    if best_f <= 1e-8:
        return best_x

    # Robust SIQI Solver
    def siqi_solve(a, b, fa, fb):
        nonlocal eval_count, best_x, best_f
        
        if fa == 0: return a
        if fb == 0: return b
        if fa * fb > 0:
            return None

        # Ensure |fa| < |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        
        for _ in range(50):
            if eval_count >= max_evals - 2:
                break
                
            if abs(fa) <= 1e-12:
                return a
            if abs(b - a) < 1e-12:
                return a
                
            x_try = None
            try:
                # Safeguarded Secant
                denom = fb - fa
                if abs(denom) > 1e-300:
                    x_secant = b - fb * (b - a) / denom
                    mid = (a + b) / 2.0
                    if a < x_secant < b:
                        # Prefer secant unless it's very close to midpoint
                        if abs(x_secant - mid) > 0.05 * abs(b - a):
                            x_try = x_secant
                        else:
                            x_try = mid
                    else:
                        x_try = mid
                else:
                    x_try = mid
            except Exception:
                x_try = mid

            fx_try = f_counted(x_try)
            
            if fx_try == 0:
                return x_try
            if abs(fx_try) < best_f:
                best_x = x_try
                best_f = abs(fx_try)
                
            if fa * fx_try < 0:
                b = x_try
                fb = fx_try
            else:
                c = a
                fc = fa
                a = x_try
                fa = fx_try
                
            if abs(fb) <= 1e-12 or abs(b - a) < 1e-12:
                return b if abs(fb) < abs(fa) else a

        return b if abs(fb) < abs(fa) else a

    def attempt_from(x_start):
        nonlocal eval_count, best_x, best_f
        
        if eval_count >= max_evals - 10:
            return False
            
        fx_start = f_counted(x_start)
        if abs(fx_start) < best_f:
            best_x = x_start
            best_f = abs(fx_start)
        if best_f <= 1e-8:
            return True
            
        # Expand search radius to find a bracket
        h = 0.1
        while h < 1e6 and eval_count < max_evals - 10:
            a = x_start - h
            b = x_start + h
            
            fa = f_counted(a)
            if abs(fa) < best_f:
                best_x = a
                best_f = abs(fa)
                if best_f <= 1e-8: return True
            
            fb = f_counted(b)
            if abs(fb) < best_f:
                best_x = b
                best_f = abs(fb)
                if best_f <= 1e-8: return True
                
            if fa * fb < 0:
                root = siqi_solve(a, b, fa, fb)
                if root is not None:
                    f_root = f_counted(root)
                    if abs(f_root) < best_f:
                        best_x = root
                        best_f = abs(f_root)
                    if best_f <= 1e-8:
                        return True
                return False 
                
            h *= 2.0
            
        return False

    # Phase 1: Local search from x0
    if attempt_from(x0):
        return best_x

    # Phase 2: Interleaved Global Search
    # Use a deterministic set of offsets to cover the space
    # Halton-like sequence or just fixed grid
    offsets = []
    # Generate offsets based on powers of 2 and fractions
    for i in range(1, 12):
        r = 2.0 ** i
        offsets.append(r)
        offsets.append(r * 0.5)
        offsets.append(r * 0.1)
        
    # Shuffle deterministically
    # Using a simple LCG for reproducibility without np.random.seed side effects
    seed = 42
    def next_rand():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) % (2**31)
        return seed / (2**31)
        
    # Create candidate points
    candidates = []
    for r in offsets:
        # Sample 2 points at this radius
        for _ in range(2):
            angle = next_rand() * 2 * math.pi
            candidates.append(x0 + r * math.cos(angle))
            candidates.append(best_x + r * math.cos(angle))
            
    # Sort by distance to x0 to try closer ones first? 
    # Or just iterate.
    
    for x_try in candidates:
        if eval_count >= max_evals - 10:
            break
        if attempt_from(x_try):
            return best_x

    # Phase 3: Fallback - Minimize f^2 around best_x
    if eval_count < max_evals - 20:
        try:
            bound = max(1.0, abs(best_x)) * 0.1
            if bound < 1e-6: bound = 1e-6
            
            res = minimize_scalar(lambda x: f_counted(x)**2, bounds=(best_x - bound, best_x + bound), 
                                   method='bounded', options={'maxiter': 50, 'xatol': 1e-12})
            
            if res.success:
                f_res = f_counted(res.x)
                if abs(f_res) < best_f:
                    best_x = res.x
                    best_f = abs(f_res)
                    
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
