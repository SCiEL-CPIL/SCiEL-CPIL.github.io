"""Improved growth index approximation using direct numerical optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

def _compute_exact_growth(Om_m0, w0, wa, a_eval):
    """
    Compute exact growth factor D(a) and growth rate f(a) = dlnD/dlna
    for a CPL cosmology using the GR ODE.
    Returns arrays of D_norm(a) and f(a) corresponding to a_eval.
    """
    # Constants
    # H(a) = H0 * sqrt(Om_m(a) + Om_de(a))
    # We work in units where H0 = 1 for the shape of D(a).
    # ODE: D'' + (1 + 1.5*Om_m(a)) * D' - 0.5*Om_m(a) * D = 0 (with respect to ln a)
    # Let x = ln(a), D(x) is growth.
    # d^2D/dx^2 + (1 + 1.5*Om_m(e^x)) * dD/dx - 0.5*Om_m(e^x) * D = 0
    
    def H_sq(a):
        # Om_m(a)
        rho_m = Om_m0 * a**-3
        # Exact CPL DE density
        # rho_de(a) = rho_de0 * exp(-3 * int_a^1 (1+w(a'))/a' da')
        # For small wa, approx a^(-3(1+w0+wa)). 
        # For better accuracy in ODE, we should use the exact integral or a good approx.
        # The standard CPL approximation for density is:
        # rho_de(a) = (1-Om_m0) * a^(-3*(1+w0+wa)) * (1 - 0.5*wa*(1-a)) ... this is messy.
        # Let's use the standard definition:
        # E^2(a) = Om_m0 a^-3 + (1-Om_m0) * exp(-3 * int_1^a (1+w(a'))/a' da')
        # int_1^a (1+w0+wa(1-a'))/a' da' = (1+w0) ln(a) + wa * (1-a)
        # So rho_de(a) = (1-Om_m0) * exp(-3 * ((1+w0) ln(a) + wa * (1-a)))
        #             = (1-Om_m0) * a^(-3(1+w0)) * exp(-3*wa*(1-a))
        
        exp_term = np.exp(-3.0 * wa * (1.0 - a))
        rho_de = (1.0 - Om_m0) * (a ** (-3.0 * (1.0 + w0))) * exp_term
        rho_m = Om_m0 * (a ** -3.0)
        return rho_m + rho_de

    def om_m(a):
        rho_m = Om_m0 * (a ** -3.0)
        exp_term = np.exp(-3.0 * wa * (1.0 - a))
        rho_de = (1.0 - Om_m0) * (a ** (-3.0 * (1.0 + w0))) * exp_term
        return rho_m / (rho_m + rho_de)

    # Define ODE in terms of x = ln(a)
    # y = [D, D']
    # D'' + (1 + 1.5*Om_m(e^x)) D' - 0.5*Om_m(e^x) D = 0
    
    def ode_func(x, y):
        a = np.exp(x)
        om = om_m(a)
        dy0 = y[1]
        dy1 = - (1.0 + 1.5 * om) * y[1] + 0.5 * om * y[0]
        return [dy0, dy1]

    # Initial conditions at a=1 (x=0)
    # Normalize D(1) = 1
    # D'(1) = f(1). We need to estimate f(1) or solve backwards/forwards.
    # Standard practice: solve from a=1 to a_min with D(1)=1, D'(1)=f0.
    # But f0 is unknown. 
    # Alternative: Solve from a_min to a=1.
    # Let's solve from x=ln(a_min) to x=0.
    # We can use a shooting method or just integrate backwards from a=1 with an initial guess for f(1) and normalize?
    # Actually, the growth factor is unique up to a constant. We can set D(1)=1.
    # We need f(1). 
    # For CPL, f(1) is close to Om_m0^gamma. 
    # Let's use a 2-point boundary value approach or just integrate from a=0.01 to 1.
    # If we integrate from a=0.01, we need initial conditions there.
    # In the matter-dominated era, D(a) ~ a. So D(0.01) ~ 0.01, D'(0.01) ~ 1.
    # Let's integrate from x_start = ln(0.01) to x_end = 0.
    
    x_start = np.log(0.01)
    x_end = 0.0
    x_eval = np.log(a_eval)
    
    # Initial conditions at x_start
    # D ~ a, so D' ~ 1.
    y0 = [0.01, 1.0]
    
    # Solve
    sol = solve_ivp(ode_func, [x_start, x_end], y0, t_eval=x_eval, rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        # Fallback to Linder approx if solver fails
        f_approx = Om_m0 ** 0.55
        return np.ones_like(a_eval), np.full_like(a_eval, f_approx)
    
    D_raw = sol.y[0]
    D_prime = sol.y[1]
    
    # Normalize D(1) = 1
    # Find index closest to x=0
    idx_1 = np.argmin(np.abs(x_eval - 0.0))
    D_norm_factor = D_raw[idx_1]
    D_norm = D_raw / D_norm_factor
    
    # f = D' / D
    f = D_prime / D_raw
    
    return D_norm, f

def _max_error(coeffs, cosmologies, a_vals):
    """
    Calculate the maximum relative error of f and D_norm for the given coefficients.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for Om_m0, w0, wa in cosmologies:
        # Get exact values
        D_exact, f_exact = _compute_exact_growth(Om_m0, w0, wa, a_vals)
        
        # Calculate approximate values
        # Need Om_m(a) for q
        # Use the same density relation as in _compute_exact_growth
        exp_term = np.exp(-3.0 * wa * (1.0 - a_vals))
        rho_de = (1.0 - Om_m0) * (a_vals ** (-3.0 * (1.0 + w0))) * exp_term
        rho_m = Om_m0 * (a_vals ** -3.0)
        Om_m_a = rho_m / (rho_m + rho_de)
        
        q = 1.0 - Om_m_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10 # Penalty for out-of-bounds gamma
            
        f_approx = Om_m_a ** gamma
        
        # D_approx = exp(-integral f dx)
        # We need to integrate f_approx from a to 1.
        # Use cumulative trapz or simple integration.
        # Since a_vals is sorted ascending, integral from a to 1 is - integral from 1 to a.
        # Let's compute integral of f(a') d(ln a') from a to 1.
        # x = ln a. dx = da/a.
        # Integral = sum (f_i + f_{i+1})/2 * (x_{i+1} - x_i)
        
        x = np.log(a_vals)
        # We want D(a)/D(1) = exp(- int_a^1 f(a') dlna')
        # Let I(a) = int_a^1 f(a') dlna'
        # I(a_i) = sum_{j=i}^{N-1} 0.5 * (f_j + f_{j+1}) * (x_{j+1} - x_j)
        
        dx = np.diff(x)
        avg_f = 0.5 * (f_approx[:-1] + f_approx[1:])
        integrand = avg_f * dx
        # Cumulative sum from the end
        # I(a_N) = 0
        # I(a_{N-1}) = integrand[N-2]
        # I(a_{N-2}) = integrand[N-2] + integrand[N-3]
        
        # Reverse integrand, cumsum, reverse back
        I = np.zeros_like(a_vals)
        I[:-1] = np.cumsum(integrand[::-1])[::-1]
        
        D_approx = np.exp(-I)
        
        # Relative errors
        # Avoid division by zero or small numbers? D and f are positive.
        err_f = np.abs(f_approx - f_exact) / np.maximum(f_exact, 1e-6)
        err_D = np.abs(D_approx - D_exact) / np.maximum(D_exact, 1e-6)
        
        local_max = np.max(np.max(err_f), np.max(err_D))
        if local_max > max_err:
            max_err = local_max
            
    return max_err

def construct():
    # Define representative cosmologies for the optimization
    # Sample a grid to cover the parameter space
    Om_m0_vals = [0.2, 0.25, 0.3]
    w0_vals = [-1.0, -0.5, 0.0]
    wa_vals = [-0.3, 0.0, 0.3]
    
    cosmologies = []
    for om in Om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                cosmologies.append((om, w0, wa))
                
    # Scale factors for evaluation
    a_vals = np.linspace(0.05, 1.0, 20)
    
    # Initial guess: Linder coefficients approximated in this basis
    # gamma ~ 0.55 + 0.05(1+w)
    # q ~ 1-Om_m, u ~ 1+w
    # If we assume c0=0.5, c3=0.05, others 0?
    # Let's start with a reasonable guess.
    x0 = [0.5, 0.0, 0.0, 0.05, 0.0, 0.0]
    
    # Minimize
    # Use Nelder-Mead as it doesn't require gradients and is robust for small dims
    result = minimize(_max_error, x0, args=(cosmologies, a_vals), method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 2000})
    
    best_coeffs = result.x
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
