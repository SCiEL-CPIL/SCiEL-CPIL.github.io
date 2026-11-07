"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import random


def construct():
    """Find optimal coefficients c0, c1, c2 to minimize the uncertainty constant."""
    
    def compute_Q_coefficients(c0, c1, c2):
        """
        Returns coefficients of Q(u) = g(x)/x^2 where u=x^2.
        Q(u) = a5*u^5 + a4*u^4 + a3*u^3 + a2*u^2 + a1*u + a0
        """
        c3 = -(c0 + 12*c1 + 1680*c2) / 4435200.0
        
        a5 = c3 * 4096.0
        a4 = c3 * (-98304.0)
        a3 = c2*256.0 + c3*1032192.0
        a2 = c2*(-3072.0) + c3*(-5308416.0)
        a1 = c1*16.0 + c2*11520.0 + c3*12096000.0
        a0 = c1*(-48.0) + c2*(-13440.0) + c3*(-12902400.0)
        
        return np.array([a5, a4, a3, a2, a1, a0])

    def objective(coeffs):
        c0, c1, c2 = coeffs
        try:
            Q_coeffs = compute_Q_coefficients(c0, c1, c2)
            
            # Ensure leading coefficient is positive (g -> +inf)
            # If a5 < 0, we flip the sign of the whole polynomial
            if Q_coeffs[0] < 0:
                Q_coeffs = -Q_coeffs
            elif Q_coeffs[0] == 0:
                # Degenerate case, handle lower degree
                # Find first non-zero
                for i in range(len(Q_coeffs)):
                    if Q_coeffs[i] != 0:
                        if Q_coeffs[i] < 0:
                            Q_coeffs = -Q_coeffs
                        break
                else:
                    return 1e100

            # Find roots of Q(u)
            roots = np.roots(Q_coeffs)
            
            # Filter for real positive roots
            real_roots = roots[np.abs(roots.imag) < 1e-9].real
            positive_roots = real_roots[real_roots > 1e-10]
            
            if len(positive_roots) == 0:
                return 1e100

            # Sort positive roots in descending order
            positive_roots.sort(reverse=True)
            
            min_r2 = float('inf')
            for r in positive_roots:
                # Check sign change
                # Since leading coeff > 0, Q(u) -> +inf as u -> inf.
                # At the largest root, if it changes sign, it must go from - to +.
                # So Q(r - eps) should be negative.
                eps = 1e-8 * max(1.0, abs(r))
                Q_left = np.polyval(Q_coeffs, r - eps)
                
                # Check if it's a simple root (derivative not zero) and sign change
                if Q_left < 0:
                    # It crosses from negative to positive. This is a valid sign change root.
                    # Since we sorted descending, the first one is the largest.
                    return r*r # We want to minimize r^2
            return 1e100
        except Exception:
            return 1e100

    # Seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Bounds
    bounds = [
        (-5.0, 5.0),    # c0
        (-1.0, 1.0),    # c1
        (-0.01, 0.01)   # c2
    ]
    
    # Multi-start Powell optimization
    best_val = float('inf')
    best_coeffs = x0
    
    # Start from seed
    res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
    if res.fun < best_val:
        best_val = res.fun
        best_coeffs = res.x

    # Random starts
    random.seed(42)
    for _ in range(10):
        start = np.array([
            random.uniform(-5.0, 5.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-0.01, 0.01)
        ])
        res = minimize(objective, start, method='Powell', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x

    # Final polish with Nelder-Mead on the best found
    res = minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_val:
        best_val = res.fun
        best_coeffs = res.x

    return best_coeffs

# EVOLVE-BLOCK-END
