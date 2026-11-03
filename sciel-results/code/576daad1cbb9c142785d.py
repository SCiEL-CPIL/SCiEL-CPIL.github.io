import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre
import time

def _compute_c4(roots):
    """Compute C4 upper bound given 12 double-root locations."""
    m = 12
    n = 1
    # P(x) = prod_{i=1}^{m} (x - r_i)^2
    # We need the Laguerre expansion of P(x) * exp(-x/2) or similar
    # C4 involves the ratio of L2 norms
    
    # Build the polynomial coefficients
    # P(x) = prod (x - r_i)^2
    poly = np.array([1.0])
    for r in roots:
        factor = np.array([1.0, -2*r, r*r])
        poly = np.convolve(poly, factor)
    
    # Now compute the Laguerre expansion
    # The C4 bound is related to:
    # C4 = sqrt( sum_{k=0}^{2m} c_k^2 * k! / (k+m)! )^{-1} or similar
    # where c_k are coefficients in the Laguerre basis
    
    # Actually, let me think about this differently.
    # In the Cohn-Goncalves construction, you have:
    # f(x) = P(x) * L_m^{(n-1)}(x) * exp(-x/2)
    # and C4 is computed from the L2 norm of f
    
    # Let me use a numerical approach:
    # The bound is C4 = 1 / ||g||_2 where g is some function
    
    # For n=1, m=12:
    # The construction gives an upper bound on the uncertainty constant
    # C4 = (integral of |f|^2)^{-1/2} or similar
    
    # Let me compute using the Laguerre coefficients
    # P(x) has degree 2m = 24
    # L_m^{(0)}(x) has degree m = 12
    # Product has degree 36
    
    # The L2 norm in the Laguerre weight:
    # ||f||_2^2 = integral_0^inf |P(x)*L_m(x)*exp(-x/2)|^2 dx
    #           = integral_0^inf P(x)^2 * L_m(x)^2 * exp(-x) dx
    
    # This can be computed via Gaussian quadrature or coefficient matching
    
    # Using the fact that Laguerre polynomials are orthogonal:
    # integral L_k(x) L_j(x) exp(-x) dx = delta_{kj} * (k!)^2
    
    # So I need to expand P(x)^2 * L_m(x)^2 in the Laguerre basis
    # and sum the squared coefficients times the norms
    
    # This is complex. Let me use numerical integration instead.
    
    # Actually, for a simpler and more direct approach:
    # C4 = 1/sqrt( integral_0^inf P(x)^2 * [L_m^{(0)}(x)]^2 * exp(-x) dx )
    
    # Let me compute this numerically
    from scipy.integrate import quad
    
    def integrand(x):
        p_val = 1.0
        for r in roots:
            p_val *= (x - r)**2
        l_val = eval_genlaguerre(m, n-1, x)
        return p_val**2 * l_val**2 * np.exp(-x)
    
    try:
        val, err = quad(integrand, 0, np.inf, limit=200)
        if val > 0:
            return 1.0 / np.sqrt(val)
        else:
            return np.inf
    except:
        return np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 600.0
    
    m = 12
    
    # Initial guess: spread roots
    x0 = np.linspace(0.5, 15.0, m)
    
    # Try optimization
    best_roots = x0.copy()
    best_c4 = _compute_c4(best_roots)
    
    def objective(x):
        return _compute_c4(x)
    
    # Use Nelder-Mead or Powell
    try:
        result = minimize(objective, x0, method='Powell', 
                         options={'maxiter': 200, 'xtol': 1e-6, 'disp': False})
        if result.fun < best_c4:
            best_c4 = result.fun
            best_roots = result.x
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 550.0:
        # Return best found
        return [float(r) for r in best_roots]
    
    # Try more iterations with Nelder-Mead
    try:
        result2 = minimize(objective, best_roots, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
        if result2.fun < best_c4:
            best_c4 = result2.fun
            best_roots = result2.x
    except:
        pass
    
    return [float(r) for r in best_roots]
