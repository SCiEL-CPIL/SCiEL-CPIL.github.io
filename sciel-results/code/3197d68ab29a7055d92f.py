"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import brentq, newton, minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Uses a robust bracketing strategy followed by Safeguarded Inverse Quadratic Interpolation (SIQI).
    Falls back to local minimization of f^2 if no sign change is detected.
    """
    x0 = float(x0)
    
    # State
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    
    # Wrapped function to count evaluations
    def f_counted(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return 0.0 # Prevent crash, though we should stop calling
        eval_count += 1
        try:
            return float(f(x))
        except Exception:
            return 0.0

    best_x = x0
    best_f = abs(f_counted(x0))
    if best_f <= tol:
        return best_x

    # Safeguarded Inverse Quadratic Interpolation (SIQI)
    def siqi_solve(a, b, fa, fb, max_iter=100):
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
        
        for _ in range(max_iter):
            if eval_count >= max_evals - 5:
                break
            
            # Update best
            if abs(fa) < best_f:
                best_x = a
                best_f = abs(fa)
            if abs(fb) < best_f:
                best_x = b
                best_f = abs(fb)
                
            if best_f <= tol:
                return best_x

            # Ensure |fa| < |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa

            if abs(fa) <= 1e-14:
                return a
            if abs(b - a) < 1e-14:
                return a

            x_try = None
            
            # Try Quadratic Interpolation (SIQI)
            # We have points (a, fa), (b, fb), (c, fc)
            if abs(b - c) > 1e-14 and abs(a - c) > 1e-14:
                try:
                    # Newton's divided differences
                    fab = (fb - fa) / (b - a)
                    fbc = (fb - fc) / (b - c)
                    fabc = (fbc - fab) / (c - a)
                    
                    # Solve fabc*(x-a)*(x-b) + fab*(x-a) + fa = 0
                    # Let t = x - a
                    # fabc * t * (t + (b-a)) + fab * t + fa = 0
                    # fabc * t^2 + (fabc*(b-a) + fab) * t + fa = 0
                    
                    A = fabc
                    B = fabc * (b - a) + fab
                    C = fa
                    
                    if abs(A) > 1e-14:
                        disc = B*B - 4*A*C
                        if disc >= 0:
                            sqrt_disc = np.sqrt(disc)
                            # Roots: (-B +/- sqrt_disc) / (2*A)
                            x1 = (-B + sqrt_disc) / (2*A)
                            x2 = (-B - sqrt_disc) / (2*A)
                            
                            # Choose root in (a, b)
                            in_range_1 = (a < x1 < b) or (b < x1 < a)
                            in_range_2 = (a < x2 < b) or (b < x2 < a)
                            
                            if in_range_1:
                                x_try = x1
                            elif in_range_2:
                                x_try = x2
                            else:
                                x_try = (a + b) / 2.0
                        else:
                            x_try = (a + b) / 2.0
                    else:
                        # Linear case
                        if abs(B) > 1e-14:
                            x_try = a - C / B
                            if not ((a < x_try < b) or (b < x_try < a)):
                                x_try = (a + b) / 2.0
                        else:
                            x_try = (a + b) / 2.0
                except Exception:
                    x_try = (a + b) / 2.0
            else:
                # Not enough points or coincident, use Secant or Bisection
                try:
                    if abs(fb - fa) > 1e-14:
                        x_secant = b - fb * (b - a) / (fb - fa)
                        if (a < x_secant < b) or (b < x_secant < a):
                            x_try = x_secant
                        else:
                            x_try = (a + b) / 2.0
                    else:
                        x_try = (a + b) / 2.0
                except Exception:
                    x_try = (a + b) / 2.0

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
                
            if abs(fb) <= 1e-14 or abs(b - a) < 1e-14:
                return b if abs(fb) < abs(fa) else a

        return b if abs(fb) < abs(fa) else a

    # Phase 1: Bracketing
    bracket_found = False
    a, b, fa, fb = None, None, None, None
    
    delta = 1.0
    for i in range(20):
        if eval_count >= max_evals - 20:
            break
            
        a = x0 - delta
        b = x0 + delta
        
        fa = f_counted(a)
        fb = f_counted(b)
        
        if abs(fa) < best_f:
            best_x = a
            best_f = abs(fa)
        if abs(fb) < best_f:
            best_x = b
            best_f = abs(fb)
            
        if best_f <= tol:
            return best_x

        if fa * fb < 0:
            bracket_found = True
            break
        
        delta *= 2.0
        if delta > 1e6:
            break

    if bracket_found:
        root = siqi_solve(a, b, fa, fb)
        if root is not None:
            f_root = f_counted(root)
            if abs(f_root) <= tol:
                return root
            if abs(f_root) < best_f:
                best_x = root
                best_f = abs(f_root)
            if eval_count < max_evals - 10:
                try:
                    root_bq = brentq(f_counted, a, b, xtol=1e-12, maxiter=50)
                    f_bq = f_counted(root_bq)
                    if abs(f_bq) < best_f:
                        best_x = root_bq
                        best_f = abs(f_bq)
                except Exception:
                    pass
            return best_x

    # Phase 2: No bracket found. Try local minimization of f^2
    if eval_count < max_evals - 20:
        center = best_x if abs(best_x - x0) > 1.0 else x0
        bound = max(1.0, abs(center) * 0.5)
        
        for scale in [1.0, 0.1, 0.01]:
            if eval_count >= max_evals - 10:
                break
            try:
                res = minimize_scalar(lambda x: f_counted(x)**2, bounds=(center - bound*scale, center + bound*scale), 
                                       method='bounded', options={'maxiter': 30, 'xatol': 1e-12})
                if res.success:
                    f_res = f_counted(res.x)
                    if abs(f_res) < best_f:
                        best_x = res.x
                        best_f = abs(f_res)
                    if best_f <= tol:
                        return best_x
            except Exception:
                pass

    return best_x
# EVOLVE-BLOCK-END
