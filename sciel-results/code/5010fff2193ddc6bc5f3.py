"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START

def _om_matter(a, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    # Om(a) = Om0 * a^-3 / (Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa)) * a^(-wa))
    # More precisely: Om(a) = Om0 * a^-3 / (Om0 * a^-3 + (1-Om0) * exp(3*integral))
    # For CPL: rho_de(a) = rho_de0 * a^(-3(1+w0+wa)) * a^(-wa) = rho_de0 * a^(-3-3w0-3wa-wa)
    # Actually: rho_de(a) = rho_de0 * a^(-3(1+w0+wa)) * a^(-wa) is wrong
    # Correct: rho_de(a) = rho_de0 * a^(-3(1+w0+wa)) * exp(-wa*ln(a)) 
    # = rho_de0 * a^(-3-3w0-3wa-wa) ... no
    # The standard CPL: w(a) = w0 + wa*(1-a)
    # rho_de(a) = rho_de0 * a^(-3(1+w0+wa)) * a^(-wa) 
    # Let me be careful: d ln rho_de / d ln a = -3(1+w)
    # w = w0 + wa(1-a)
    # integral from 1 to a of -3(1+w0+wa(1-a')) da'/a'
    # = -3(1+w0) ln a - 3wa * integral_1^a (1-a')/a' da'
    # = -3(1+w0) ln a - 3wa * [a' - ln a']_1^a ... wait
    # integral (1-a')/a' da' = integral (1/a' - 1) da' = ln a' - a'
    # from 1 to a: (ln a - a) - (0 - 1) = ln a - a + 1
    # So rho_de(a) = rho_de0 * exp(-3(1+w0) ln a - 3wa(ln a - a + 1))
    # = rho_de0 * a^(-3(1+w0)) * exp(-3wa(ln a - a + 1))
    
    Om0 = 0.3
    if a <= 0:
        return 1.0
    x = np.log(a)
    exponent = -3.0*(1.0+w0)*x - 3.0*wa*(x - a + 1.0)
    rho_de = np.exp(exponent)
    rho_m = Om0 * a**(-3.0) / (1.0 - Om0)  # normalized so that rho_m(1)+rho_de(1)=1
    # Actually let's normalize: at a=1, rho_m=Om0, rho_de=1-Om0
    # rho_m(a) = Om0 * a^-3
    # rho_de(a) = (1-Om0) * a^(-3(1+w0+wa)) * a^(-wa) ... 
    # Let me just use: rho_de(a)/(1-Om0) = exp(-3*integral_1^a (1+w(a')) da'/a')
    # integral_1^a (1+w0+wa(1-a')) da'/a' = (1+w0)*ln(a) + wa*(a-1) - wa*ln(a)
    # = (1+w0-wa)*ln(a) + wa*(a-1)
    # So rho_de(a) = (1-Om0) * exp(-(1+w0-wa)*3*ln(a) - 3*wa*(a-1))
    # = (1-Om0) * a^(-3(1+w0-wa)) * exp(-3*wa*(a-1))
    
    rho_m = Om0 * a**(-3.0)
    rho_de = (1.0 - Om0) * a**(-3.0*(1.0+w0-wa)) * np.exp(-3.0*wa*(a-1.0))
    total = rho_m + rho_de
    return rho_m / total

def _gamma_from_coeffs(coeffs, a, w0, wa):
    """Compute gamma from the 6 coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _om_matter(a, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa*(1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_approx(coeffs, a, w0, wa):
    """Compute f = Om(a)^gamma using the approximate gamma."""
    Om = _om_matter(a, w0, wa)
    gamma = _gamma_from_coeffs(coeffs, a, w0, wa)
    # Clamp gamma to [0.1, 1.2]
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om**gamma

def _growth_ode_reference(a_arr, w0, wa):
    """Solve the GR ODE for the growth factor to get reference f and D."""
    # D'' + 2H D' - 4πG ρ_m D = 0
    # In terms of x = ln a: d²D/dx² + (3/2 + f_H) dD/dx - (3/2) Om D = 0
    # where f_H = (1/2) d ln H²/dx
    # H²(a) = H0² [Om a^-3 + (1-Om) a^(-3(1+w0+wa)) * a^(-wa) ... ]
    # Let me use: H²(a)/H0² = Om a^-3 + (1-Om) * exp(-3*integral_1^a (1+w) da'/a')
    # = Om a^-3 + (1-Om) * a^(-3(1+w0-wa)) * exp(-3wa(a-1))
    
    def H2(a):
        return Om0 * a**(-3.0) + (1.0-Om0) * a**(-3.0*(1.0+w0-wa)) * np.exp(-3.0*wa*(a-1.0))
    
    Om0 = 0.3
    
    # Use x = ln a, so a = exp(x)
    # d²D/dx² + (3/2 + (1/2)*d ln H²/dx) dD/dx - (3/2) Om(a) D = 0
    # d ln H²/dx = (1/H²) * dH²/da * da/dx = (1/H²) * dH²/da * a
    
    def ode_system(x, y):
        a = np.exp(x)
        D, Dp = y
        H2_val = H2(a)
        # dH2/da = Om0*(-3)*a^-4 + (1-Om0)*[(-3*(1+w0-wa))*a^(-3*(1+w0-wa)-1) * exp(-3wa(a-1)) + a^(-3*(1+w0-wa)) * exp(-3wa(a-1)) * (-3wa)]
        dH2_da = (Om0 * (-3.0) * a**(-4.0) + 
                  (1.0-Om0) * (a**(-3.0*(1.0+w0-wa)) * np.exp(-3.0*wa*(a-1.0))) * 
                  (-3.0*(1.0+w0-wa)/a - 3.0*wa))
        dlnH2_dx = dH2_da * a / H2_val
        Om_a = _om_matter(a, w0, wa)
        Dpp = -1.5 * Om_a * D - (1.5 + 0.5*dlnH2_dx) * Dp
        return [Dp, Dpp]
    
    # Solve from a=1 (x=0) backwards to a_min
    x0 = 0.0
    # At a=1: D=1, D' = f(a=1) = Om0^gamma... but we don't know gamma exactly
    # Use D(1)=1, and D'(1) ≈ 0.55 (constant gamma approximation) as initial condition
    # Actually, for the reference, we should use the exact solution
    # Let's integrate forward from a small value to a=1, then normalize
    # Or integrate backward from a=1
    
    a_min = 0.05
    x_min = np.log(a_min)
    
    # Integrate from x=0 to x_min (backward)
    sol = solve_ivp(ode_system, [0.0, x_min], [1.0, 0.55], 
                    method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    
    # The reference f = D'/D
    # The reference D normalized: D(a)/D(1)
    f_ref = np.zeros_like(a_arr)
    D_ref = np.zeros_like(a_arr)
    for i, a in enumerate(a_arr):
        x = np.log(a)
        if sol.success:
            y = sol.sol(x)
            D_val = y[0]
            Dp_val = y[1]
            D_ref[i] = D_val  # D(1)=1 by construction
            f_ref[i] = Dp_val / D_val if abs(D_val) > 1e-15 else 0.55
        else:
            D_ref[i] = 1.0
            f_ref[i] = 0.55
    
    return f_ref, D_ref

def _compute_max_relative_error(coeffs, test_cosmos):
    """Compute the maximum relative error over all test cosmologies."""
    max_err = 0.0
    a_values = np.linspace(0.1, 1.0, 50)
    
    for w0, wa, Om0_test in test_cosmos:
        # Temporarily override Om0 for this cosmology
        global Om0
        old_Om0 = Om0
        Om0 = Om0_test
        
        # Get reference
        f_ref, D_ref = _growth_ode_reference(a_values, w0, wa)
        
        # Get approximation
        f_approx = np.array([_f_approx(coeffs, a, w0, wa) for a in a_values])
        
        # Compute D_approx
        # D(a)/D(1) = exp(-integral from a to 1 of f(exp(x)) dx) where x=ln(a)
        # = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        D_approx = np.ones_like(a_values)
        for i, a in enumerate(a_values):
            if a >= 1.0:
                D_approx[i] = 1.0
            else:
                x_arr = np.linspace(np.log(a), 0.0, 50)
                f_vals = np.array([_f_approx(coeffs, np.exp(x), w0, wa) for x in x_arr])
                integral = np.trapz(f_vals, x_arr)
                D_approx[i] = np.exp(-integral)
        
        # Relative errors
        mask_f = np.abs(f_ref) > 1e-8
        if np.any(mask_f):
            rel_err_f = np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])
            max_err = max(max_err, np.max(rel_err_f))
        
        mask_D = np.abs(D_ref) > 1e-8
        if np.any(mask_D):
            rel_err_D = np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D])
            max_err = max(max_err, np.max(rel_err_D))
        
        Om0 = old_Om0
    
    return max_err

Om0 = 0.3  # Global for use in helper functions

def construct():
    """Find optimal 6 coefficients for the growth-index formula."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Test cosmologies: (w0, wa, Om0)
    test_cosmos = [
        (-1.0, 0.0, 0.3),    # ΛCDM
        (-1.0, 0.5, 0.3),    # w0=-1, wa=0.5
        (-0.5, -0.5, 0.3),   # quintessence-like
        (-1.0, 1.0, 0.3),    # strong wa
        (-0.9, 0.3, 0.25),   # different Om0
        (-1.1, 0.2, 0.35),   # phantom-like
        (-1.0, 0.0, 0.25),   # different Om0, ΛCDM
        (-1.0, 0.0, 0.35),   # different Om0, ΛCDM
    ]
    
    # Start with constant gamma seed
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_error = _compute_max_relative_error(best_coeffs, test_cosmos)
    
    def objective(coeffs):
        coeffs = np.clip(coeffs, -10.0, 10.0)
        # Check gamma range constraint
        a_check = np.linspace(0.1, 1.0, 20)
        for w0, wa, Om0_t in test_cosmos:
            for a in a_check:
                g = _gamma_from_coeffs(coeffs, a, w0, wa)
                if g < 0.1 or g > 1.2:
                    return 10.0  # Penalty
        return _compute_max_relative_error(coeffs, test_cosmos)
    
    # Stage 1: Nelder-Mead from seed
    result = minimize(objective, best_coeffs, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
    if result.fun < best_error:
        best_error = result.fun
        best_coeffs = np.clip(result.x, -10.0, 10.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_coeffs.tolist()
    
    # Stage 2: Try a few different starting points
    start_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.55, 0.05, 0.05, 0.05, 0.05, 0.05],
        [0.50, 0.1, 0.0, 0.1, 0.0, 0.0],
        [0.60, -0.05, 0.0, 0.05, 0.0, 0.0],
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(objective, np.array(sp), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_error:
                best_error = res.fun
                best_coeffs = np.clip(res.x, -10.0, 10.0)
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_coeffs.tolist()
    
    # Stage 3: Refine with smaller steps
    try:
        res = minimize(objective, best_coeffs, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                               'x0': best_coeffs})
        if res.fun < best_error:
            best_error = res.fun
            best_coeffs = np.clip(res.x, -10.0, 10.0)
    except Exception:
        pass
    
    return best_coeffs.tolist()

# EVOLVE-BLOCK-END
