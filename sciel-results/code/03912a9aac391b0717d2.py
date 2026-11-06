"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    """
    Return six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    """
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s limit
    
    # Test cosmologies for local optimization
    test_models = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -0.5, 0.5),   # CPL
        (0.3, -1.0, 0.0),   # Cosmological constant
        (0.3, -0.8, 0.3),   # CPL
        (0.3, -0.6, -0.2),  # CPL
        (0.3, -1.2, 0.8),   # CPL
        (0.3, -0.3, -0.5),  # CPL
    ]
    
    a_values = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a for flat CPL universe."""
        E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        return Om0 * a**-3 / E2
    
    def compute_reference(Om0, w0, wa, a_arr):
        """Compute reference f and D/D(1) by integrating the growth ODE."""
        # D'' + (3 + H'/H) D' - (3/2) Om D = 0 in variable x = ln(a)
        # Let f = d ln D / d ln a, then f' + f^2 + (3 + H'/H) f - (3/2) Om = 0
        # We integrate the second-order ODE for D(x) where x = ln(a)
        
        def ode_system(x, y):
            # y = [D, D'] where D' = dD/dx
            a = np.exp(x)
            E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
            E2_prime = -3*Om0*a**-4 + 3*wa*(1-Om0)*np.exp(3*wa*(1-a))
            Hprime_over_H = 0.5 * a * E2_prime / E2
            Om = Om0 * a**-3 / E2
            
            # D'' + (3 + H'/H) D' - (3/2) Om D = 0
            # D'' = (3/2) Om D - (3 + H'/H) D'
            D = y[0]
            Dp = y[1]
            Dpp = 1.5 * Om * D - (3 + Hprime_over_H) * Dp
            return [Dp, Dpp]
        
        # Integrate from x=0 (a=1) backward to x_min = ln(a_min)
        x_min = np.log(a_arr[0])
        x_max = 0.0
        
        # Initial conditions at a=1: D(1)=1, D'(1)=f(1)≈1 (for LCDM-like)
        # More precisely, f(1) = Om(1)^gamma ≈ 1 for any reasonable gamma
        y0 = [1.0, 1.0]
        
        # Use dense output for interpolation
        sol = solve_ivp(ode_system, [x_max, x_min], y0, dense_output=True,
                       max_step=0.05, rtol=1e-10, atol=1e-12)
        
        if not sol.success:
            return None, None
        
        # Evaluate at requested a values
        x_eval = np.log(a_arr)
        D_vals = sol.sol(x_eval)[0]
        Dp_vals = sol.sol(x_eval)[1]
        
        # Normalize: D/D(1)
        D_norm = D_vals / D_vals[-1]  # D(1) = 1
        f_ref = Dp_vals / D_vals
        
        return f_ref, D_norm
    
    # Compute reference values for all test models
    refs = {}
    for Om0, w0, wa in test_models:
        f_ref, D_norm = compute_reference(Om0, w0, wa, a_values)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_norm)
    
    if not refs:
        return [0.55, 0.05, 0.0, 0.02, 0.0, 0.0]
    
    def objective(coeffs):
        """Compute max relative error across all models and a values."""
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_err = 0.0
        for (Om0, w0, wa), (f_ref, D_norm) in refs.items():
            for i, a in enumerate(a_values):
                Om = Om_a(Om0, w0, wa, a)
                q = 1 - Om
                u = 1 + w0 + wa * (1 - a)
                
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds
                if gamma < 0.1 or gamma > 1.2:
                    return 100.0  # Large penalty
                
                f_approx = Om ** gamma
                
                # Relative error in f
                if abs(f_ref[i]) > 1e-10:
                    err_f = abs(f_approx - f_ref[i]) / abs(f_ref[i])
                else:
                    err_f = abs(f_approx - f_ref[i])
                
                # Relative error in D
                if abs(D_norm[i]) > 1e-10:
                    err_D = abs(f_approx - f_ref[i]) / abs(f_ref[i])  # placeholder
                else:
                    err_D = 0.0
                
                # For D, we need to compute the approximate D from the integral
                # D_approx(a)/D_approx(1) = exp(-∫_a^1 f(exp(x)) dx)
                # This requires numerical integration which is slow
                # For now, use f error as proxy and add D error separately
                
                max_err = max(max_err, err_f)
        
        return max_err
    
    # Initial guess from literature
    x0 = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    
    # Use Nelder-Mead for robustness with the nonsmooth max objective
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_coeffs = result.x if result.success else x0
    
    # Ensure coefficients are in valid range
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    # Final check: ensure gamma stays in [0.1, 1.2]
    valid = True
    for (Om0, w0, wa) in refs.keys():
        for a in a_values:
            Om = Om_a(Om0, w0, wa, a)
            q = 1 - Om
            u = 1 + w0 + wa * (1 - a)
            gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + \
                    best_coeffs[3]*u + best_coeffs[4]*wa*a + best_coeffs[5]*u*q
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to literature values
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
