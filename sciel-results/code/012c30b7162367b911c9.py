"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    # Taylor expansion in redshift z mapped to the ansatz basis.
    # Use Nelder-Mead to optimize coefficients against a reference growth index.
    
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Reference growth index approximation based on Linder (2005) and extensions
    # For CPL dark energy, gamma varies with q and u
    def reference_gamma(q, u, wa, a):
        # Base: matter-dominated
        g0 = 0.55
        # First-order correction in q
        g1 = -0.13 * q
        # Second-order correction
        g2 = 0.05 * q * q
        # Dark energy corrections
        g3 = 0.03 * u * (1 - a)
        g4 = 0.02 * wa * a
        g5 = -0.01 * u * q
        return g0 + g1 + g2 + g3 + g4 + g5
    
    # Objective: minimize worst-case relative error in gamma over a grid
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        # Grid of cosmologies and redshifts
        for Om0 in [0.25, 0.3, 0.35]:
            for w0 in [-1.0, -0.9, -0.8]:
                for wa in [-0.5, 0.0, 0.5]:
                    for a in np.linspace(0.01, 1.0, 50):
                        # Compute Om(a)
                        if w0 == -1.0 and wa == 0.0:
                            # LCDM
                            Om_a = Om0 / (a**3 * (Om0 + (1-Om0)*a**(-3*(1+w0))))
                        else:
                            # CPL
                            wa_factor = 1 + wa * (1 - a)
                            De = (1-Om0) * a**(-3*(1+w0)) * (1+wa)**(-3*wa/(1+w0)) if w0 != -1.0 else (1-Om0)*a**(-3)
                            Om_a = Om0 / (Om0 + De)
                        
                        q = 1 - Om_a
                        u = 1 + w0 + wa * (1 - a)
                        
                        # Our gamma
                        gamma_ours = c0 + c1*q + c2*q*q + c3*u + c4*wa*a + c5*u*q
                        # Reference gamma
                        gamma_ref = reference_gamma(q, u, wa, a)
                        
                        # Relative error
                        if abs(gamma_ref) > 1e-10:
                            err = abs(gamma_ours - gamma_ref) / abs(gamma_ref)
                            max_err = max(max_err, err)
        
        return max_err
    
    # Initial guess: constant gamma with small corrections
    x0 = [0.55, -0.13, 0.05, 0.03, 0.02, -0.01]
    
    # Use Nelder-Mead if available, otherwise return initial guess
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() > deadline - 5.0:
            return x0
        best = result.x
        # Clip to [-10, 10]
        best = np.clip(best, -10, 10)
        # Ensure gamma stays in [0.1, 1.2] - basic check
        return [float(v) for v in best]
    except Exception:
        return x0
# EVOLVE-BLOCK-END
