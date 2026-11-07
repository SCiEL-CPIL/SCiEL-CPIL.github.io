import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Strategy:
    1. Check if x0 is already a root.
    2. Try to reconstruct the polynomial using Lagrange interpolation at 4 points.
       Solve for roots analytically using numpy.roots.
       Select the root nearest to x0.
    3. Verify the root. If it fails, fall back to bracketing + Newton.
    """
    TOL = 1e-8
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)

    # Strategy 1: Polynomial Reconstruction (4 calls)
    # Assume f is a cubic (or lower). We evaluate at 4 points to get coefficients.
    # Points: x0, x0 + h, x0 - h, x0 + 2h (or similar)
    # Actually, to be safe with numerical stability, use distinct points.
    # Let's use x0, x0+1, x0-1, x0+2? Or scaled?
    # Since we don't know the scale, let's try a few scales.
    
    # The task says "3 fixed smooth polynomial equations". It doesn't specify degree.
    # However, "Scalar root finder on 3 fixed smooth polynomial equations" often implies low degree.
    # If degree > 3, 4 points won't reconstruct it.
    # But the directive says: "Replace the iterative refinement with a single-shot polynomial reconstruction: evaluate f at 4 distinct points to determine the cubic coefficients... requiring only 4 function calls per equation."
    # This implies the hidden functions ARE cubics (or we are gambling on it).
    # If they are higher degree, this will fail and we must fall back.
    # If they are cubics, this is very fast (4 calls + 1 verification = 5 calls).
    # Current score 1.0296 suggests mean calls are low, but we can potentially lower them further.
    # Current method uses ~5-10 calls typically. 5 calls is a slight improvement if it works.
    # But if it fails, we waste 4 calls and then use the old method.
    # Let's try this. If it fails, we fall back.
    
    # We need to pick 4 points.
    # If x0 is close to root, points should be close.
    # Let's try points: x0, x0+1, x0-1, x0+2.
    # If the polynomial has large coefficients, this might be unstable.
    # But let's try.
    
    # To improve stability, we can use points clustered around x0.
    # h = 1.0
    # Points: x0 - 1.5, x0 - 0.5, x0 + 0.5, x0 + 1.5
    
    # Let's try a few sets of points to increase chance of success?
    # No, that increases calls. The directive says "single-shot".
    # So we pick ONE set.
    
    # If the root is far from x0, interpolation might be poor.
    # But we verify.
    
    # Let's use points: x0, x0+1, x0-1, x0+2
    # Or better, symmetric: x0-1.5, x0-0.5, x0+0.5, x0+1.5
    
    try:
        h = 1.0
        x1 = x0 - 1.5 * h
        x2 = x0 - 0.5 * h
        x3 = x0 + 0.5 * h
        x4 = x0 + 1.5 * h
        
        f1 = f(x1)
        f2 = f(x2)
        f3 = f(x3)
        f4 = f(x4)
        
        if all(math.isfinite(v) for v in [f1, f2, f3, f4]):
            # Solve for coefficients of p(x) = a*x^3 + b*x^2 + c*x + d
            # Using Lagrange or Vandermonde
            # Vandermonde matrix V where V[i][j] = x_i^j
            # V * [d, c, b, a]^T = [f1, f2, f3, f4]^T
            # Note: numpy.roots expects [a, b, c, d]
            
            # Check if points are distinct
            if len(set([x1, x2, x3, x4])) < 4:
                # Degenerate, skip
                pass
            else:
                V = np.vander([x1, x2, x3, x4], N=4, increasing=True)
                # V is 4x4
                # Solve V * coeff_vec = f_vec
                # coeff_vec = [d, c, b, a]
                try:
                    coeff_vec = np.linalg.solve(V, np.array([f1, f2, f3, f4]))
                    # coeff_vec = [d, c, b, a]
                    a, b, c, d = coeff_vec[3], coeff_vec[2], coeff_vec[1], coeff_vec[0]
                    
                    # If a is very small, it might be quadratic or lower.
                    # numpy.roots handles this.
                    roots = np.roots([a, b, c, d])
                    
                    # Find real roots
                    real_roots = []
                    for r in roots:
                        if abs(r.imag) < 1e-10:
                            real_roots.append(r.real)
                    
                    if real_roots:
                        # Select root nearest to x0
                        best_r = min(real_roots, key=lambda r: abs(r - x0))
                        
                        # Verify
                        f_best = f(best_r)
                        if math.isfinite(f_best) and abs(f_best) <= TOL:
                            return best_r
                        # If verification fails, it's not a cubic or points were bad.
                        # Fall through to other methods.
                except np.linalg.LinAlgError:
                    pass
    except Exception:
        pass

    # Strategy 2: Bracketing + Brent's Method (Fallback)
    step = 1.0
    for i in range(100):
        if not check_deadline():
            break
            
        a = x0 - step
        b = x0 + step
        
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
            
        if fa == 0:
            return a
        if fb == 0:
            return b
            
        if fa * fb < 0:
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                break
                
        step *= 2.0
        if step > 1e10:
            break

    # Strategy 3: Safeguarded Newton
    x = x0
    fx = fx0
    
    for i in range(50):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        h = 1e-5 * max(1.0, abs(x))
        
        try:
            fx_h = f(x + h)
            fx_mh = f(x - h)
        except Exception:
            break
            
        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break
            
        dfx = (fx_h - fx_mh) / (2 * h)
        
        if abs(dfx) < 1e-15:
            break
            
        step = fx / dfx
        x_new = x - step
        
        if not is_finite(x_new) or abs(step) > 10 * max(1.0, abs(x)):
            damp = 0.5
            for _ in range(10):
                x_damped = x - damp * step
                if is_finite(x_damped):
                    try:
                        fx_damped = f(x_damped)
                    except Exception:
                        damp *= 0.5
                        continue
                    if is_finite(fx_damped):
                        if abs(fx_damped) < best_f_abs:
                            best_f_abs = abs(fx_damped)
                            best_x = x_damped
                        if abs(fx_damped) <= TOL:
                            return x_damped
                        if abs(fx_damped) < abs(fx):
                            x = x_damped
                            fx = fx_damped
                            break
                damp *= 0.5
            if abs(fx) >= best_f_abs:
                break
            continue
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            break
            
        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new
            
        if abs(fx_new) <= TOL:
            return x_new
            
        if abs(fx_new) > abs(fx) * 1.1 and i > 2:
            damp = 0.5
            improved = False
            for _ in range(10):
                x_damped = x - damp * (x_new - x)
                if is_finite(x_damped):
                    try:
                        fx_damped = f(x_damped)
                    except Exception:
                        damp *= 0.5
                        continue
                    if is_finite(fx_damped):
                        if abs(fx_damped) < abs(fx):
                            x = x_damped
                            fx = fx_damped
                            improved = True
                            break
                damp *= 0.5
            if not improved:
                break
            continue
            
        x = x_new
        fx = fx_new

    return best_x
# EVOLVE-BLOCK-END
