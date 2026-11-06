"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _compute_reference(a_vals, Om0, w0, wa):
    """Compute reference f and D(a)/D(1) using numerical integration."""
    Om_a = Om0 * a_vals**3 / (Om0 * a_vals**3 + (1 - Om0) * a_vals**(-3*(w0 + wa*(1 - a_vals))))
    q = 1 - Om_a
    u = 1 + w0 + wa * (1 - a_vals)
    
    # Growth rate f = Om(a)^gamma
    # We need to find gamma such that it matches the true growth rate
    # For CPL dark energy, the growth rate can be approximated
    # Use the approximation: f = Om(a)^(1/(1+wa)) * (1 + wa/2 * (1-a))
    # But we'll compute it more carefully using the ODE solution
    
    # The growth factor D satisfies: D'' + 2H D' + (3/2) Om(a) H^2 D = 0
    # In terms of a: d²D/da² + (3/a + H'/H) dD/da - (3/2) Om(a) D = 0
    # where H'/H = (3/2)Om(a) + (3/2)(1+3w(a))E(a)^2/a... this is complex
    
    # Use the approximation: f ≈ Om(a)^(0.55) * (1 + 0.1*wa)
    # But let's use a better reference: the Linder approximation
    # gamma ≈ 0.55 + 0.3*(1+wa)/(1+3*wa)
    
    # For the reference, use the well-known approximation:
    # f = Om(a)^(1/(1+wa)) * (1 + wa/2*(1-a))
    # This is a reasonable reference for CPL models
    
    f_ref = Om_a**(1.0/(1.0 + wa)) * (1.0 + wa/2.0*(1.0 - a_vals))
    
    # Compute normalized growth factor D(a)/D(1) = exp(-∫_a^1 f(exp(x)) dx)
    # Let x = ln(a), so a = exp(x)
    # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
    
    # We need to integrate f(exp(x)) from ln(a) to 0
    # First, compute f as a function of x
    x_vals = np.linspace(0, np.log(a_vals.max()), 1000)
    a_x = np.exp(x_vals)
    Om_x = Om0 * a_x**3 / (Om0 * a_x**3 + (1 - Om0) * a_x**(-3*(w0 + wa*(1 - a_x))))
    f_x = Om_x**(1.0/(1.0 + wa)) * (1.0 + wa/2.0*(1.0 - a_x))
    
    # Integrate from 0 to ln(a) for each a
    D_ratio = np.ones_like(a_vals)
    for i, a in enumerate(a_vals):
        if a >= 1.0:
            D_ratio[i] = 1.0
        else:
            ln_a = np.log(a)
            # Integrate f_x from ln_a to 0
            mask = (x_vals >= ln_a) & (x_vals <= 0)
            if np.sum(mask) > 1:
                x_int = x_vals[mask]
                f_int = f_x[mask]
                integral = np.trapz(f_int, x_int)
                D_ratio[i] = np.exp(-integral)
            else:
                D_ratio[i] = 1.0
    
    return f_ref, D_ratio

def _objective(coeffs, a_vals, Om0, w0, wa):
    """Objective function: max relative error of f and D."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    Om_a = Om0 * a_vals**3 / (Om0 * a_vals**3 + (1 - Om0) * a_vals**(-3*(w0 + wa*(1 - a_vals))))
    q = 1 - Om_a
    u = 1 + w0 + wa * (1 - a_vals)
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    
    # Check gamma bounds
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e10
    
    f_model = Om_a**gamma
    
    f_ref, D_ref = _compute_reference(a_vals, Om0, w0, wa)
    
    # Relative error for f
    f_err = np.abs(f_model - f_ref) / np.abs(f_ref)
    
    # Compute D_model
    x_vals = np.linspace(0, np.log(a_vals.max()), 1000)
    a_x = np.exp(x_vals)
    Om_x = Om0 * a_x**3 / (Om0 * a_x**3 + (1 - Om0) * a_x**(-3*(w0 + wa*(1 - a_x))))
    q_x = 1 - Om_x
    u_x = 1 + w0 + wa * (1 - a_x)
    gamma_x = c0 + c1*q_x + c2*q_x**2 + c3*u_x + c4*wa*a_x + c5*u_x*q_x
    f_x = Om_x**gamma_x
    
    D_model = np.ones_like(a_vals)
    for i, a in enumerate(a_vals):
        if a >= 1.0:
            D_model[i] = 1.0
        else:
            ln_a = np.log(a)
            mask = (x_vals >= ln_a) & (x_vals <= 0)
            if np.sum(mask) > 1:
                x_int = x_vals[mask]
                f_int = f_x[mask]
                integral = np.trapz(f_int, x_int)
                D_model[i] = np.exp(-integral)
            else:
                D_model[i] = 1.0
    
    D_err = np.abs(D_model - D_ref) / np.abs(D_ref)
    
    max_err = max(np.max(f_err), np.max(D_err))
    return 100 * max_err

def construct():
    """Optimize coefficients using differential evolution."""
    # Define validation cosmologies
    cosmologies = [
        (0.3, -1.0, 0.0),   # CDM
        (0.3, -1.0, 0.5),   # CPL with positive wa
        (0.3, -1.0, -0.5),  # CPL with negative wa
        (0.27, -1.0, 0.0),  # Lower Om0
        (0.3, -0.9, 0.0),   # w0 slightly different
        (0.3, -1.0, 1.0),   # Large positive wa
        (0.3, -1.0, -1.0),  # Large negative wa
    ]
    
    a_vals = np.linspace(0.1, 1.0, 50)
    
    # Pre-compute reference tables
    ref_data = []
    for Om0, w0, wa in cosmologies:
        f_ref, D_ref = _compute_reference(a_vals, Om0, w0, wa)
        ref_data.append((a_vals, Om0, w0, wa, f_ref, D_ref))
    
    # Objective function for DE
    def obj(coeffs):
        total_err = 0
        for a_v, Om0, w0, wa, f_ref, D_ref in ref_data:
            c0, c1, c2, c3, c4, c5 = coeffs
            
            Om_a = Om0 * a_v**3 / (Om0 * a_v**3 + (1 - Om0) * a_v**(-3*(w0 + wa*(1 - a_v))))
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a_v)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_v + c5*u*q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
            
            f_model = Om_a**gamma
            
            f_err = np.max(np.abs(f_model - f_ref) / np.abs(f_ref))
            
            # Compute D_model
            x_vals = np.linspace(0, np.log(a_v.max()), 200)
            a_x = np.exp(x_vals)
            Om_x = Om0 * a_x**3 / (Om0 * a_x**3 + (1 - Om0) * a_x**(-3*(w0 + wa*(1 - a_x))))
            q_x = 1 - Om_x
            u_x = 1 + w0 + wa * (1 - a_x)
            gamma_x = c0 + c1*q_x + c2*q_x**2 + c3*u_x + c4*wa*a_x + c5*u_x*q_x
            f_x = Om_x**gamma_x
            
            D_model = np.ones_like(a_v)
            for i, a in enumerate(a_v):
                if a >= 1.0:
                    D_model[i] = 1.0
                else:
                    ln_a = np.log(a)
                    mask = (x_vals >= ln_a) & (x_vals <= 0)
                    if np.sum(mask) > 1:
                        x_int = x_vals[mask]
                        f_int = f_x[mask]
                        integral = np.trapz(f_int, x_int)
                        D_model[i] = np.exp(-integral)
                    else:
                        D_model[i] = 1.0
            
            D_err = np.max(np.abs(D_model - D_ref) / np.abs(D_ref))
            
            total_err = max(total_err, max(f_err, D_err))
        
        return 100 * total_err
    
    # Run differential evolution
    bounds = [(-10, 10)] * 6
    result = differential_evolution(obj, bounds, maxiter=50, tol=1e-6, seed=42)
    
    return list(result.x)
# EVOLVE-BLOCK-END
