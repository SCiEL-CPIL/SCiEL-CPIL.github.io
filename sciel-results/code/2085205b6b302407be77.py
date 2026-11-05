"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize

# Pre-compute reference data
# We will use a grid of w0, wa
W0_GRID = [-1.2, -1.0, -0.8]
WA_GRID = [-0.5, 0.0, 0.5]
OM_M0 = 0.3
A_GRID = np.linspace(0.1, 1.0, 20)

def H2(a, w0, wa, om_m):
    om_de = 1.0 - om_m
    term_m = om_m * a**(-3.0)
    term_de = om_de * a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
    return term_m + term_de

def get_om_m(a, w0, wa, om_m0):
    h2 = H2(a, w0, wa, om_m0)
    return om_m0 * a**(-3.0) / h2

def compute_single_reference(w0, wa, om_m0, a_grid):
    def ode(t, y):
        a = np.exp(t)
        d = y[0]
        dp = y[1]
        
        om_m = get_om_m(a, w0, wa, om_m0)
        h2 = H2(a, w0, wa, om_m0)
        
        om_de = 1.0 - om_m0
        c1 = om_m0
        c2 = om_de
        
        h2_prime = -3.0 * c1 * a**(-4.0)
        k = -3.0 * (1.0 + w0)
        g = -3.0 * wa * (1.0 - a)
        g_prime = 3.0 * wa
        term_de_prime = c2 * (k * a**(k-1.0) * np.exp(g) + a**k * np.exp(g) * g_prime)
        h2_prime += term_de_prime
        
        d_ln_h2_d_ln_a = a * h2_prime / h2
        coeff1 = 3.0/(2.0*a) + 0.5 * d_ln_h2_d_ln_a
        coeff2 = 1.5 * om_m / (a**2)
        
        dpp = -coeff1 * dp + coeff2 * d
        return [dp, dpp]
        
    a_min = min(a_grid)
    a_max = max(a_grid)
    t_span = (np.log(a_min), np.log(a_max))
    y0 = [a_min**(5.0/3.0), (5.0/3.0) * a_min**(2.0/3.0)]
    
    sol = solve_ivp(ode, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=False)
    
    D = sol.y[0]
    Dp = sol.y[1]
    a_vals = np.exp(sol.t)
    
    D = D / D[-1]
    Dp = Dp / D[-1]
    
    f_ref = a_vals * Dp / D
    D_norm_ref = D
    
    return f_ref, D_norm_ref, a_vals

def construct():
    # 1. Pre-compute references
    refs = []
    for w0 in W0_GRID:
        for wa in WA_GRID:
            f_ref, D_ref, a_ref = compute_single_reference(w0, wa, OM_M0, A_GRID)
            refs.append((w0, wa, f_ref, D_ref, a_ref))
            
    def loss(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for w0, wa, f_ref, D_ref, a_ref in refs:
            om_m = np.array([get_om_m(ai, w0, wa, OM_M0) for ai in a_ref])
            q = 1.0 - om_m
            u = 1.0 + w0 + wa * (1.0 - a_ref)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_ref + c5*u*q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
                
            f_ans = om_m**gamma
            
            x = np.log(a_ref)
            integrand = np.log(f_ans)
            ln_D_rel = cumulative_trapezoid(integrand, x, initial=0)
            D_ans = np.exp(ln_D_rel)
            
            err_f = np.abs(f_ans - f_ref) / np.abs(f_ref)
            err_D = np.abs(D_ans - D_ref) / np.abs(D_ref)
            
            err_f = np.clip(err_f, 0, 100)
            err_D = np.clip(err_D, 0, 100)
            
            local_max = np.max(np.concatenate([err_f, err_D]))
            max_err = max(max_err, local_max)
            
        return max_err

    x0 = [0.55, 0.15, -0.05, -0.5, 0.3, 0.1]
    
    res = minimize(loss, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 1000, 'maxfev': 1000})
    
    if res.fun < 1e10:
        coeffs = np.clip(res.x, -10, 10)
        return coeffs
    else:
        return x0

# EVOLVE-BLOCK-END
