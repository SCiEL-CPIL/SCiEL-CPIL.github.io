"""Root finding via BFGS optimization on g(x)=f(x)^2 with Brent fallback for ill-conditioned problems."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses BFGS on g(x)=f(x)^2 as primary method, with Brent's method as fallback.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    
    eval_count = [0]
    best_x = float(x0)
    best_fx_abs = float('inf')
    
    def safe_f(x):
        if eval_count[0] >= max_evals:
            return 0.0
        eval_count[0] += 1
        try:
            val = f(x)
            if not isinstance(val, (int, float)):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return 1e100
            return val
        except Exception:
            return 1e100
    
    def track(x, fx):
        nonlocal best_x, best_fx_abs
        if abs(fx) < best_fx_abs:
            best_fx_abs = abs(fx)
            best_x = x
    
    x = float(x0)
    fx = safe_f(x)
    track(x, fx)
    if abs(fx) <= 1e-8:
        return x
    
    # First try to find a bracket - if found, use Brent (most reliable)
    a = x
    fa = fx
    step = max(1e-4, abs(x) * 1e-4)
    bracket_found = False
    b = None
    fb = None
    
    for direction in [1, -1]:
        step = max(1e-4, abs(x) * 1e-4)
        for _ in range(60):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            b = x + direction * step
            fb = safe_f(b)
            track(b, fb)
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                bracket_found = True
                break
            step *= 1.8
            if step > 1e12:
                break
        if bracket_found:
            break
    
    if bracket_found:
        # Use Brent's method on the bracket [a, b]
        a_val, b_val = a, b
        fa_val, fb_val = fa, fb
        
        if abs(fa_val) > abs(fb_val):
            a_val, b_val = b_val, a_val
            fa_val, fb_val = fb_val, fa_val
        
        c_val = a_val
        fc_val = fa_val
        d_val = b_val - a_val
        e_val = d_val
        
        for _ in range(200):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            if abs(fa_val) <= 1e-8:
                return a_val
            if abs(fb_val) <= 1e-8:
                return b_val
            
            # Bisection point
            m_val = 0.5 * (a_val + b_val)
            e_val = 0.5 * (a_val - b_val)
            
            if abs(m_val) < 1e-14 and abs(fa_val) < 1e-14:
                return a_val
            
            # Try inverse quadratic interpolation
            if abs(e_val) >= 1e-14 and abs(d_val) >= 1e-14:
                if abs(fa_val) != abs(fb_val):
                    # IQU
                    p_val = a_val + (a_val - b_val) * fa_val / (fb_val - fa_val) * (b_val - a_val) / (a_val - b_val)
                    # Standard IQU formula
                    if fb_val != fa_val:
                        ratio = fa_val / fb_val
                        if ratio != 0 and ratio != 1:
                            p_val = m_val + (fa_val - fb_val) * fa_val / (fb_val - fa_val)
                            # Simpler: p = a - fa*(b-a)/(fb-fa)
                            if fb_val != fa_val:
                                p_val = a_val - fa_val * (b_val - a_val) / (fb_val - fa_val)
                                lo = min(a_val, b_val)
                                hi = max(a_val, b_val)
                                if lo < p_val < hi:
                                    fp_val = safe_f(p_val)
                                    track(p_val, fp_val)
                                    if abs(fp_val) <= 1e-8:
                                        return p_val
                                    if fp_val * fa_val < 0:
                                        b_val, fb_val = p_val, fp_val
                                    else:
                                        a_val, fa_val = p_val, fp_val
                                    c_val, fc_val = b_val, fb_val
                                    d_val = b_val - a_val
                                    e_val = d_val
                                    continue
            
            # Secant
            if abs(d_val) >= 1e-14:
                if fb_val != fa_val:
                    p_val = a_val - fa_val * (b_val - a_val) / (fb_val - fa_val)
                    lo = min(a_val, b_val)
                    hi = max(a_val, b_val)
                    if lo < p_val < hi:
                        fp_val = safe_f(p_val)
                        track(p_val, fp_val)
                        if abs(fp_val) <= 1e-8:
                            return p_val
                        if fp_val * fa_val < 0:
                            b_val, fb_val = p_val, fp_val
                        else:
                            a_val, fa_val = p_val, fp_val
                        c_val, fc_val = b_val, fb_val
                        d_val = b_val - a_val
                        e_val = d_val
                        continue
            
            # Bisection
            fp_val = safe_f(m_val)
            track(m_val, fp_val)
            if abs(fp_val) <= 1e-8:
                return m_val
            if fp_val * fa_val < 0:
                b_val, fb_val = m_val, fp_val
            else:
                a_val, fa_val = m_val, fp_val
            c_val, fc_val = b_val, fb_val
            d_val = b_val - a_val
            e_val = d_val
        
        if abs(fa_val) <= 1e-8:
            return a_val
        if abs(fb_val) <= 1e-8:
            return b_val
        if abs(fa_val) < abs(fb_val):
            return a_val
        return b_val
    
    # No bracket found - use BFGS on g(x) = f(x)^2
    # BFGS for 1D: maintain Hessian approximation H (scalar)
    # g(x) = f(x)^2
    # g'(x) = 2*f(x)*f'(x)  -- but we don't have f'(x) directly
    # Use finite differences for gradient
    
    # BFGS state: x, g, grad, H (1x1 matrix = scalar)
    x_curr = x
    fx_curr = fx
    g_curr = fx_curr * fx_curr
    
    # Initial Hessian estimate
    h = 1e-6 * max(1.0, abs(x_curr))
    fx_plus = safe_f(x_curr + h)
    track(x_curr + h, fx_plus)
    g_plus = fx_plus * fx_plus
    grad_curr = (g_plus - g_curr) / h
    
    # BFGS Hessian approximation (1D: just a scalar)
    H = 1.0  # Initial guess
    
    for iteration in range(100):
        if time.time() > deadline or eval_count[0] >= max_evals:
            break
        
        if g_curr <= 1e-16:  # |f| <= 1e-8
            return x_curr
        
        # Compute search direction: p = -H * grad
        p = -H * grad_curr
        
        if abs(p) < 1e-30:
            break
        
        # Line search: backtracking
        alpha = 1.0
        for _ in range(30):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            x_trial = x_curr + alpha * p
            fx_trial = safe_f(x_trial)
            track(x_trial, fx_trial)
            g_trial = fx_trial * fx_trial
            
            # Armijo condition: g_trial <= g_curr + c1 * alpha * grad * p
            c1 = 1e-4
            if g_trial <= g_curr + c1 * alpha * grad_curr * p:
                break
            alpha *= 0.5
        
        if alpha < 1e-10:
            # Line search failed, try smaller step
            x_trial = x_curr + 0.01 * p
            fx_trial = safe_f(x_trial)
            track(x_trial, fx_trial)
            g_trial = fx_trial * fx_trial
            alpha = 0.01
        
        x_new = x_curr + alpha * p
        fx_new = safe_f(x_new)
        track(x_new, fx_new)
        g_new = fx_new * fx_new
        
        # Compute new gradient
        h2 = 1e-6 * max(1.0, abs(x_new))
        fx_new_plus = safe_f(x_new + h2)
        track(x_new + h2, fx_new_plus)
        g_new_plus = fx_new_plus * fx_new_plus
        grad_new = (g_new_plus - g_new) / h2
        
        # BFGS update: s = x_new - x_curr, y = grad_new - grad_curr
        s = x_new - x_curr
        y = grad_new - grad_curr
        
        if abs(s) > 1e-15:
            sy = s * y
            if sy > 1e-15:  # Curvature condition
                # 1D BFGS: H_new = H - s^2 * y / (s*y) + ... 
                # In 1D, BFGS simplifies. The update for H^{-1} (inverse Hessian):
                # Actually for 1D, we can just use the secant condition:
                # H_new * grad_new ≈ -p_new
                # But simpler: update H to satisfy secant: H_new * y = s
                # So H_new = s / y (if y != 0)
                if abs(y) > 1e-15:
                    H_new = s / y
                    # Damp if H becomes too large or negative
                    if H_new > 0 and H_new < 1e10:
                        H = H_new
                    else:
                        H = max(1e-10, min(H, 1e10))
                else:
                    H = max(1e-10, min(H, 1e10))
        
        x_curr = x_new
        fx_curr = fx_new
        g_curr = g_new
        grad_curr = grad_new
        
        if abs(fx_curr) <= 1e-8:
            return x_curr
    
    return best_x
# EVOLVE-BLOCK-END
