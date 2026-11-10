"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad
import time

# EVOLVE-BLOCK-START
def _compute_om(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    w_int = w0 * (1 - a) + wa * (1 - a)**2 / 2.0
    de_factor = (1.0 - Om0) * np.exp(3.0 * w_int)
    m_factor = Om0 * a**3
    return m_factor / (m_factor + de_factor)

def _gamma_of(a, c, Om0, w0, wa):
    """Compute gamma(a) = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    Om_a = _compute_om(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_of(a, c, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)"""
    Om_a = _compute_om(a, Om0, w0, wa)
    g = _gamma_of(a, c, Om0, w0, wa)
    return Om_a**g

def _normalized_D(a, c, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from a to 1 of f(exp(x)) dx) where x = ln(a)"""
    if a >= 1.0:
        return 1.0
    
    x0 = np.log(a)
    
    def integrand(x):
        a_val = np.exp(x)
        return _f_of(a_val, c, Om0, w0, wa)
    
    integral, _ = quad(integrand, x0, 0.0, limit=200, epsabs=1e-10, epsrel=1e-10)
    return np.exp(-integral)

def _solve_growth_ode(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=500):
    """
    Solve the growth ODE numerically to get reference f(a) and D(a)/D(1).
    
    The growth equation in terms of x = ln(a):
    d²D/dx² + [2 + H'(a)/H(a)] dD/dx - (3/2) Om_m(a) D = 0
    
    For CPL dark energy in a flat universe:
    H(a) = H0 * sqrt(Om0*a^3 + (1-Om0)*exp(-3*(w0*(1-a) + wa*(1-a)^2/2)))
    H'/H = (1/(2*H^2)) * d(H^2)/da * da/dx = (1/(2*H^2)) * d(H^2)/da * a
    
    Let's define:
    E(a)^2 = Om0*a^3 + (1-Om0)*exp(-3*(w0*(1-a) + wa*(1-a)^2/2))
    H = H0 * E(a)
    
    d(H^2)/da = 3*Om0*a^2 + (1-Om0)*exp(-3*w_int) * (-3) * d(w_int)/da
    where w_int = w0*(1-a) + wa*(1-a)^2/2
    d(w_int)/da = -w0 - wa*(1-a)
    
    So d(H^2)/da = 3*Om0*a^2 + 3*(1-Om0)*exp(-3*w_int) * (w0 + wa*(1-a))
    
    H'/H = (1/(2*E^2)) * d(E^2)/da * a
    """
    
    a_vals = np.linspace(a_min, a_max, n_points)
    x_vals = np.log(a_vals)
    
    # Compute E^2 and its derivative
    def E2(a):
        w_int = w0 * (1 - a) + wa * (1 - a)**2 / 2.0
        return Om0 * a**3 + (1 - Om0) * np.exp(-3 * w_int)
    
    def dE2_da(a):
        w_int = w0 * (1 - a) + wa * (1 - a)**2 / 2.2
        dw_int_da = -w0 - wa * (1 - a)
        return 3 * Om0 * a**2 + 3 * (1 - Om0) * np.exp(-3 * w_int) * dw_int_da
    
    # Convert ODE to first-order system in x = ln(a)
    # Let y = [D, D'] where D' = dD/dx
    # D'' + [2 + H'/H] D' - (3/2) Om_m(a) D = 0
    # D'' = -[2 + H'/H] D' + (3/2) Om_m(a) D
    
    def ode_system(x, y):
        a = np.exp(x)
        E2_val = E2(a)
        dE2_val = dE2_da(a)
        H_prime_over_H = (a * dE2_val) / (2 * E2_val)
        Om_a = Om0 * a**3 / E2_val
        
        D, D_prime = y
        D_double_prime = -[2 + H_prime_over_H] * D_prime + (3/2) * Om_a * D
        return [D_prime, D_double_prime]
    
    # Integrate from x=ln(a_max)=0 to x=ln(a_min)
    x_max = 0.0
    x_min = np.log(a_min)
    
    # Initial conditions at a=1: D(1)=1, D'(1) = f(1) ≈ Om0^0.55
    D_init = 1.0
    D_prime_init = Om0**0.55
    
    x_grid = np.linspace(x_min, x_max, n_points)
    
    try:
        sol = solve_ivp(ode_system, [x_min, x_max], [D_init, D_prime_init], 
                       t_eval=x_grid, method='RK45', rtol=1e-10, atol=1e-12)
        
        if not sol.success:
            # Fallback to approximation
            f_ref = np.array([_compute_om(a, Om0, w0, wa)**(0.55 + 0.05*(w0 - 0.5*wa)) for a in a_vals])
            D_ref = np.exp(-np.array([quad(lambda x: _compute_om(np.exp(x), Om0, w0, wa)**(0.55 + 0.05*(w0 - 0.5*wa)), np.log(av), 0.0)[0] for av in a_vals]))
            return a_vals, f_ref, D_ref
        
        D_vals = sol.y[0]
        D_prime_vals = sol.y[1]
        f_vals = D_prime_vals / D_vals
        
        # Normalize D so that D(1) = 1
        # Find the index closest to x=0
        idx_max = np.argmin(np.abs(x_grid - 0.0))
        D_norm = D_vals / D_vals[idx_max]
        
        return a_vals, f_vals, D_norm
    
    except Exception:
        # Fallback to approximation
        f_ref = np.array([_compute_om(a, Om0, w0, wa)**(0.55 + 0.05*(w0 - 0.5*wa)) for a in a_vals])
        D_ref = np.exp(-np.array([quad(lambda x: _compute_om(np.exp(x), Om0, w0, wa)**(0.55 + 0.05*(w0 - 0.5*wa)), np.log(av), 0.0)[0] for av in a_vals]))
        return a_vals, f_ref, D_ref

def _interpolate_reference(a_query, a_ref, f_ref, D_ref):
    """Interpolate reference values at query points."""
    f_interp = np.interp(a_query, a_ref, f_ref)
    D_interp = np.interp(a_query, a_ref, D_ref)
    return f_interp, D_interp

def _objective_joint(c, test_cosmos, a_test, ref_cache):
    """Compute the maximum relative error across all test cosmologies."""
    max_err = 0.0
    
    for (Om0, w0, wa), (a_ref, f_ref, D_ref) in zip(test_cosmos, ref_cache):
        for i, a in enumerate(a_test):
            # Get reference values
            f_ref_val, D_ref_val = _interpolate_reference(a, a_ref, f_ref, D_ref)
            
            if f_ref_val is not None and abs(f_ref_val) > 1e-10:
                f_approx = _f_of(a, c, Om0, w0, wa)
                err_f = abs(f_approx - f_ref_val) / abs(f_ref_val)
                max_err = max(max_err, err_f)
            
            if D_ref_val is not None and abs(D_ref_val) > 1e-10:
                D_approx = _normalized_D(a, c, Om0, w0, wa)
                err_D = abs(D_approx - D_ref_val) / abs(D_ref_val)
                max_err = max(max_err, err_D)
    
    return max_err

def construct():
    """Find optimal coefficients for the growth-index formula."""
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # Test cosmologies: (Om0, w0, wa)
    # Use a range of plausible values
    test_cosmos = [
        (0.3, -1.0, 0.0),      # Lambda CDM
        (0.3, -1.0, 0.5),      # CPL with positive wa
        (0.3, -0.5, -1.0),     # CPL with negative wa
        (0.25, -1.0, 0.3),     # Lower Om0
        (0.35, -1.0, -0.5),    # Higher Om0
    ]
    
    # Test scale factors
    a_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    # Pre-compute reference solutions for all test cosmologies
    print(f"Pre-computing references... (time: {time.time()-start_time:.1f}s)")
    ref_cache = []
    for Om0, w0, wa in test_cosmos:
        a_ref, f_ref, D_ref = _solve_growth_ode(Om0, w0, wa)
        ref_cache.append((a_ref, f_ref, D_ref))
        print(f"  Reference for ({Om0}, {w0}, {wa}) done (time: {time.time()-start_time:.1f}s)")
    
    # Initial guess: constant gamma = 0.55
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0.copy()
    best_score = _objective_joint(c0, test_cosmos, a_test, ref_cache)
    print(f"Initial score: {best_score:.6f} (time: {time.time()-start_time:.1f}s)")
    
    # Use Nelder-Mead with joint objective
    def obj(c):
        return _objective_joint(c, test_cosmos, a_test, ref_cache)
    
    # First pass: coarse optimization
    if time.time() < deadline - 30.0:
        try:
            result = minimize(obj, c0, method='Nelder-Mead', 
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_c = result.x.copy()
                print(f"After coarse optimization: {best_score:.6f} (time: {time.time()-start_time:.1f}s)")
        except Exception as e:
            print(f"Coarse optimization error: {e}")
    
    # Second pass: fine optimization
    if time.time() < deadline - 15.0:
        try:
            result = minimize(obj, best_c, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_c = result.x.copy()
                print(f"After fine optimization: {best_score:.6f} (time: {time.time()-start_time:.1f}s)")
        except Exception as e:
            print(f"Fine optimization error: {e}")
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Ensure gamma stays in [0.1, 1.2] for test cases
    # If not, fall back to seed
    valid = True
    for Om0, w0, wa in test_cosmos:
        for a in a_test:
            g = _gamma_of(a, best_c, Om0, w0, wa)
            if g < 0.1 or g > 1.2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        print("Gamma constraint violated, falling back to seed")
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print(f"Final score: {best_score:.6f} (time: {time.time()-start_time:.1f}s)")
    return list(best_c)
# EVOLVE-BLOCK-END
