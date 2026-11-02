import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return poly(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    # We need ∫|P(x)|*exp(-x²/2)dx = ∫|P(x)|*exp(x²/2)*exp(-x²)dx
    # Using Gauss-Hermite with weight exp(-x²):
    # ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
    # So ∫|P(x)|*exp(-x²/2)dx ≈ Σ w_i |P(x_i)|*exp(x_i²/2)
    n_gh = 100
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)
    
    def objective_fast(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            p_vals = poly(x_gh, c0, c4, c8)
            abs_f = np.abs(p_vals) * np.exp(x_gh**2 / 2.0)
            integral = np.sum(w_gh * abs_f)
            return (2 * integral) ** 2
        except Exception:
            return np.inf
    
    # Smooth surrogate: soft-abs via log-sum-exp to enable gradient-based methods
    def objective_smooth(coeffs, smooth_param=0.05):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            p_vals = poly(x_gh, c0, c4, c8)
            weighted = p_vals * np.exp(x_gh**2 / 2.0) * w_gh
            # Soft absolute value: softabs(x) = sqrt(x^2 + eps^2)
            soft_abs = np.sqrt(weighted**2 + smooth_param**2)
            integral = np.sum(soft_abs)
            return (2 * integral) ** 2
        except Exception:
            return np.inf
    
    # Gradient of the smooth objective for use with BFGS/L-BFGS-B
    def objective_smooth_grad(coeffs, smooth_param=0.05):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf, np.zeros(3)
        try:
            p_vals = poly(x_gh, c0, c4, c8)
            weighted = p_vals * np.exp(x_gh**2 / 2.0) * w_gh
            soft_abs = np.sqrt(weighted**2 + smooth_param**2)
            integral = np.sum(soft_abs)
            val = (2 * integral) ** 2
            
            # Gradient via chain rule
            # d(soft_abs)/d(weighted) = weighted / sqrt(weighted^2 + eps^2)
            d_soft_abs_dw = weighted / soft_abs
            # d(weighted)/d(coeffs) involves derivatives of poly
            # poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
            # c12 = -(c0 + 3*c4 + 105*c8)/10395
            # dpoly/dc0 = H0 - H12/10395
            # dpoly/dc4 = H4 - 3*H12/10395
            # dpoly/dc8 = H8 - 105*H12/10395
            H0v = H0(x_gh)
            H4v = H4(x_gh)
            H8v = H8(x_gh)
            H12v = H12(x_gh)
            
            dpoly_dc0 = H0v - H12v / 10395.0
            dpoly_dc4 = H4v - 3*H12v / 10395.0
            dpoly_dc8 = H8v - 105*H12v / 10395.0
            
            scale = np.exp(x_gh**2 / 2.0) * w_gh
            
            grad_c0 = 4 * integral * np.sum(d_soft_abs_dw * scale * dpoly_dc0)
            grad_c4 = 4 * integral * np.sum(d_soft_abs_dw * scale * dpoly_dc4)
            grad_c8 = 4 * integral * np.sum(d_soft_abs_dw * scale * dpoly_dc8)
            
            return val, np.array([grad_c0, grad_c4, grad_c8])
        except Exception:
            return np.inf, np.zeros(3)
    
    # Verify with quad for the initial guess
    def objective_quad(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            def abs_f(x):
                return abs(f_of_x(x, c0, c4, c8))
            val, _ = integrate.quad(abs_f, 0, np.inf, limit=200)
            return (2 * val) ** 2
        except Exception:
            return np.inf
    
    # Homotopy continuation: start from a known two-term solution and gradually
    # introduce H8 and H12 contributions to navigate the non-convex landscape
    
    # Step 1: Find optimal two-term solution (c0, c4) with c8=0
    # When c8=0, c12 = -(c0 + 3*c4)/10395
    # This is a simpler problem we can solve more reliably
    
    # Analytic starting point: pure Gaussian (c0=1, c4=0, c8=0) gives c12 = -1/10395
    # But we need g(0)=0, which is already enforced by the c12 constraint
    
    # Start with the seed program's initial guess
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = objective_fast(x0)
    best_coeffs = x0.copy()
    
    # Homotopy path: gradually increase the weight of H8 and H12 terms
    # Parameter t in [0, 1]: at t=0, use only H0, H4 (c8=0, c12 from H0,H4)
    # at t=1, use full H0, H4, H8, H12
    
    def objective_homotopy(coeffs, t):
        """Objective with homotopy parameter t controlling H8/H12 contribution"""
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            # Scale H8 and H12 contributions by t
            # Effective coefficients: c0, c4, t*c8, and c12 adjusted accordingly
            c8_eff = t * c8
            c12 = -(c0 + 3*c4 + 105*c8_eff) / 10395.0
            p_vals = c0*H0(x_gh) + c4*H4(x_gh) + c8_eff*H8(x_gh) + c12*H12(x_gh)
            abs_f = np.abs(p_vals) * np.exp(x_gh**2 / 2.0)
            integral = np.sum(w_gh * abs_f)
            return (2 * integral) ** 2
        except Exception:
            return np.inf
    
    # Phase 1: Optimize with t=0 (two-term problem) to get a reliable starting point
    if time.time() < deadline - 40:
        # Optimize c0, c4 with c8=0
        def obj_two_term(params):
            c0, c4 = params
            if time.time() > deadline - 5:
                return np.inf
            try:
                c8_eff = 0.0
                c12 = -(c0 + 3*c4) / 10395.0
                p_vals = c0*H0(x_gh) + c4*H4(x_gh) + c12*H12(x_gh)
                abs_f = np.abs(p_vals) * np.exp(x_gh**2 / 2.0)
                integral = np.sum(w_gh * abs_f)
                return (2 * integral) ** 2
            except Exception:
                return np.inf
        
        # Multiple restarts for the two-term problem
        rng = np.random.RandomState(42)
        best_two_term_score = np.inf
        best_two_term_coeffs = np.array([-1.13, 0.04])
        
        for restart in range(50):
            if time.time() > deadline - 25:
                break
            if restart == 0:
                start = np.array([-1.13, 0.04])
            else:
                start = best_two_term_coeffs + rng.normal(0, 0.1, 2)
            try:
                res = opt.minimize(obj_two_term, start, method='Nelder-Mead',
                                  options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300,
                                         'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_two_term_score:
                    best_two_term_score = res.fun
                    best_two_term_coeffs = res.x.copy()
            except Exception:
                pass
        
        # Use this as starting point for homotopy
        homotopy_start = np.array([best_two_term_coeffs[0], best_two_term_coeffs[1], 0.0])
        
    else:
        homotopy_start = x0.copy()
    
    # Phase 2: Homotopy continuation from t=0 to t=1
    n_homotopy_steps = 20
    current_coeffs = homotopy_start.copy()
    
    for step in range(n_homotopy_steps):
        if time.time() > deadline - 20:
            break
        t = step / (n_homotopy_steps - 1) if step < n_homotopy_steps - 1 else 1.0
        t_next = (step + 1) / (n_homotopy_steps - 1) if step < n_homotopy_steps - 1 else 1.0
        
        # Optimize at current t
        def obj_at_t(params, t_val=t):
            return objective_homotopy(params, t_val)
        
        try:
            res = opt.minimize(obj_at_t, current_coeffs, method='Nelder-Mead',
                              options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200,
                                     'adaptive': True})
            if np.isfinite(res.fun):
                current_coeffs = res.x.copy()
        except Exception:
            pass
        
        # Evaluate at full problem (t=1) and update best
        if step >= n_homotopy_steps - 3:  # Near the end
            score = objective_fast(current_coeffs)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_coeffs = current_coeffs.copy()
    
    # Phase 3: Full optimization with multiple restarts using fast objective
    rng = np.random.RandomState(42)
    
    for restart in range(100):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            x_start = best_coeffs.copy()
        elif restart == 1:
            x_start = homotopy_start.copy()
        else:
            # Random restart around best with decreasing scale
            scale = 0.3 * (1 + 5.0/(restart + 1.0))
            x_start = best_coeffs + rng.normal(0, scale, 3)
        
        try:
            res = opt.minimize(objective_fast, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Gradient-based refinement using smooth surrogate
    if time.time() < deadline - 10:
        for smooth_param in [0.1, 0.05, 0.01, 0.005]:
            if time.time() > deadline - 5:
                break
            try:
                res = opt.minimize(objective_smooth, best_coeffs, method='BFGS',
                                  jac=lambda c: objective_smooth_grad(c, smooth_param)[1],
                                  options={'gtol': 1e-10, 'maxiter': 200})
                if np.isfinite(res.fun):
                    # Verify with exact objective
                    exact_score = objective_fast(res.x)
                    if np.isfinite(exact_score) and exact_score < best_score:
                        best_score = exact_score
                        best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Final polish with high-precision quad
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(objective_quad, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # One more polish pass with even tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective_quad, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
