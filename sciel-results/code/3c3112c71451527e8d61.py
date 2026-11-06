"""Direct approach for ill-conditioned root finding with multiple strategies."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    max_evals = 500
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def secant_method(x0, tol=TOL, max_iter=200):
        """Secant method with careful handling of ill-conditioning."""
        x1 = float(x0)
        h = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h
        
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                h2 = 1e-4 * max(1.0, abs(x2)) * (1 + 0.1 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False
            
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        
        return abs(f1) <= tol or abs(f2) <= tol

    def newton_forward(x0, tol=TOL, max_iter=100):
        """Newton's method with forward finite difference for derivative."""
        x = float(x0)
        
        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True
            
            h = 1e-6 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
            except Exception:
                return False
            
            d = (fp - fx) / h
            if abs(d) < 1e-30:
                h2 = 1e-3 * max(1.0, abs(x))
                try:
                    fp2 = call_f(x + h2)
                    d = (fp2 - fx) / h2
                except Exception:
                    return False
                if abs(d) < 1e-30:
                    return False
            
            step = fx / d
            max_step = 5.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x = x_new
        
        try:
            fx = call_f(x)
        except Exception:
            return False
        return abs(fx) <= tol

    def bisection(x0, tol=TOL, max_iter=200):
        """Bisection with bracket search."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True
        
        step = 1e-3 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None
        
        for i in range(40):
            x_test = x + step * (2 ** i)
            try:
                f_test = call_f(x_test)
            except Exception:
                break
            if not math.isfinite(f_test):
                break
            if fa is None:
                a, fa = x, fx
            if fa * f_test < 0:
                b, fb = x_test, f_test
                break
        
        if b is None:
            a, b = None, None
            fa, fb = None, None
            for i in range(40):
                x_test = x - step * (2 ** i)
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if fb is None:
                    b, fb = x, fx
                if fb * f_test < 0:
                    a, fa = x_test, f_test
                    break
        
        if a is None or b is None:
            return False
        
        for i in range(max_iter):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= tol:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        
        return min(abs(fa), abs(fb)) <= tol

    def bfgs_scalar(x0, tol=TOL, max_iter=150):
        """
        BFGS-style curvature-aware step for minimizing g(x) = f(x)^2.
        In 1D, BFGS maintains an estimate of g'(x) and the inverse Hessian H.
        The step is s = -H * g'(x). We update H using the BFGS formula:
        H_new = H - (H*s*y^T + y*s^T*H)/(y^T*H*y) + (1 + y^T*H*y)/(y^T*y)^2 * y*y^T
        In 1D this simplifies significantly.
        """
        x = float(x0)
        scale = 1e-4 * max(1.0, abs(x))
        
        try:
            f0 = call_f(x)
        except Exception:
            return False
        if abs(f0) <= tol:
            return True
        
        # Initial derivative estimate via central difference
        h = 1e-5 * max(1.0, abs(x))
        try:
            fp = call_f(x + h)
            fm = call_f(x - h)
        except Exception:
            return False
        
        # g(x) = f(x)^2, g'(x) = 2*f(x)*f'(x)
        f_prime = (fp - fm) / (2 * h)
        g_prime = 2.0 * f0 * f_prime
        g_val = f0 * f0
        
        if abs(g_prime) < 1e-30:
            return False
        
        # Initial inverse Hessian estimate: H = 1 / g''(x)
        # g''(x) = 2*(f'(x)^2 + f(x)*f''(x))
        # Approximate f''(x) via second difference
        f_pp = (fp - 2*f0 + fm) / (h * h)
        g_pp = 2.0 * (f_prime * f_prime + f0 * f_pp)
        
        if abs(g_pp) < 1e-30:
            H = 1.0 / max(g_prime * g_prime, 1e-30) * 0.01  # small initial H
        else:
            H = 1.0 / g_pp
            # Clamp H to reasonable range
            if not math.isfinite(H) or abs(H) < 1e-20:
                H = 0.01 / max(1.0, abs(x))
            elif abs(H) > 1e10:
                H = math.copysign(1e10, H)
        
        # Ensure H is positive (for minimization)
        if H <= 0:
            H = abs(H) * 0.1 + 1e-10
        
        prev_x = x
        prev_g_prime = g_prime
        prev_H = H
        
        for i in range(max_iter):
            if not check_time():
                return False
            
            # BFGS step: s = -H * g_prime
            s = -H * g_prime
            
            # Damping: limit step size
            max_step = 5.0 * max(1.0, abs(x))
            if abs(s) > max_step:
                s = math.copysign(max_step, s)
            
            x_new = x + s
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            try:
                f_new = call_f(x_new)
            except Exception:
                return False
            
            if abs(f_new) <= tol:
                return True
            
            g_new = f_new * f_new
            
            # Estimate new gradient
            h2 = 1e-5 * max(1.0, abs(x_new))
            try:
                fp_new = call_f(x_new + h2)
                fm_new = call_f(x_new - h2)
            except Exception:
                # Fallback to forward difference
                try:
                    fp_new = call_f(x_new + h2)
                    fm_new = f_new
                    h2 = h2
                except Exception:
                    return False
            
            f_prime_new = (fp_new - fm_new) / (2 * h2)
            g_prime_new = 2.0 * f_new * f_prime_new
            
            if abs(g_prime_new) < 1e-30:
                # Gradient nearly zero, check if we're at a minimum
                if g_new <= tol * tol:
                    return True
                return False
            
            # BFGS update for H (1D case)
            # s_vec = x_new - x, y_vec = g_prime_new - g_prime
            s_vec = x_new - prev_x
            y_vec = g_prime_new - prev_g_prime
            
            if abs(s_vec) > 1e-15 and abs(y_vec) > 1e-15:
                sy = s_vec * y_vec
                if sy > 0:  # Curvature condition
                    Hy = prev_H * y_vec
                    yHy = y_vec * Hy
                    y2 = y_vec * y_vec
                    # H_new = H - (H*s*y + y*s*H)/(y*H*y) + (1 + y*H*y)/(y^2)^2 * y*y
                    # In 1D: H_new = H * (1 - s*y/(y*H*y)) + (1 + y*H*y)/(y^2)
                    # Simplified: H_new = H * y^2 / (y^2 - s*y*H) + (1 + y*H*y)/(y^2)
                    # Let's use the standard 1D BFGS update:
                    # H_new = H - (H^2 * s * y) / (y * H * y) + (1 + y*H*y) * y / (y^2)
                    # = H - H^2*s*y/(y*H*y) + (1+y*H*y)/(y)
                    # Wait, let me re-derive for 1D:
                    # H_new = H - (H*s*y^T + y*s^T*H)/(y^T*H*y) + (1 + y^T*H*y)/(y^T*y)^2 * y*y^T
                    # In 1D: H_new = H - (2*H*s*y)/(y*H*y) + (1 + y*H*y)/(y*y)^2 * y*y
                    # = H - 2*H*s*y/(y*H*y) + (1 + y*H*y)/(y^2)
                    # = H - 2*s/(y) + (1 + y*H*y)/(y^2)
                    # Hmm, that doesn't look right. Let me use the simpler form:
                    # For 1D, the BFGS update is:
                    # H_new = H * (1 - rho * s * y) + (1 + rho * y * H * y) * (s^2 / (y^2))
                    # where rho = 1/(s*y)
                    # Actually, the standard 1D BFGS update simplifies to:
                    # H_new = H - (H*s*y)/(y*H*y) * 2 + ... 
                    # Let me just use the inverse Hessian update directly:
                    # H_new = H * (1 - s*y/(y*H*y)) + (1 + y*H*y)/(y^2)
                    # But this can be unstable. Let's use a safer approach:
                    # Simply update H based on the observed curvature:
                    # H_new = H * (g_prime / g_prime_new)^2 if signs agree
                    # Or use the secant-based update: H_new = s / y_vec
                    # The secant update H = s/y is actually the 1D BFGS when H_old is ignored
                    H_new = s_vec / y_vec
                    if not math.isfinite(H_new) or abs(H_new) < 1e-20:
                        H_new = prev_H
                    elif abs(H_new) > 1e10:
                        H_new = math.copysign(1e10, H_new)
                    if H_new <= 0:
                        H_new = abs(H_new) * 0.5 + 1e-10
                    H = H_new
                else:
                    # Curvature condition violated, keep H
                    pass
            else:
                # Can't update, keep H
                pass
            
            prev_x = x
            prev_g_prime = g_prime
            prev_H = H
            x = x_new
            g_prime = g_prime_new
        
        return abs(f_new) <= tol

    def polish_pass(x0, tol=TOL, max_iter=50):
        """High-precision polish using secant with very tight tolerance."""
        x1 = float(x0)
        h = 1e-8 * max(1.0, abs(x1))
        x2 = x1 + h
        
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                return False
            
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
            
            if abs(f2) <= tol:
                return True
        
        return abs(f1) <= tol or abs(f2) <= tol

    # Try strategies in order of expected efficiency
    # 1. Secant (fastest convergence for smooth functions)
    if secant_method(x0):
        return best_x
    
    # 2. BFGS-style curvature-aware method (good for ill-conditioned)
    if bfgs_scalar(x0):
        return best_x
    
    # 3. Newton with forward difference
    if newton_forward(x0):
        return best_x
    
    # 4. Bisection (robust but slow)
    if bisection(x0):
        return best_x
    
    # 5. If we have a decent best_x, try to polish it
    if best_fabs < 1e-4:
        polish_pass(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
