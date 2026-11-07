"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import sympy as sp


def construct():
    """Find optimal coefficients c0, c1, c2 to minimize the uncertainty constant."""
    
    # Use sympy for exact-ish polynomial construction
    x = sp.symbols('x')
    
    # Get Hermite polynomials
    H0 = sp.hermite(0, x)
    H4 = sp.hermite(4, x)
    H8 = sp.hermite(8, x)
    H12 = sp.hermite(12, x)
    
    # Expand them to get coefficients
    H0_exp = sp.expand(H0)
    H4_exp = sp.expand(H4)
    H8_exp = sp.expand(H8)
    H12_exp = sp.expand(H12)
    
    # Extract coefficients for powers of x
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 98304x^10 + 1032192x^8 - 5308416x^6 + 12096000x^4 - 12902400x^2 + 4435200
    
    # Let's work with the polynomial in u = x^2
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g(0) = c0*1 + c1*12 + c2*1680 + c3*4435200 = 0
    # So c3 = -(c0 + 12*c1 + 1680*c2) / 4435200
    
    # g(x) as polynomial in x:
    # Power 12: c3 * 4096
    # Power 10: c3 * (-98304)
    # Power 8: c2*256 + c3*1032192
    # Power 6: c2*(-3072) + c3*(-5308416)
    # Power 4: c1*16 + c2*11520 + c3*12096000
    # Power 2: c1*(-48) + c2*(-13440) + c3*(-12902400)
    # Power 0: c0 + c1*12 + c2*1680 + c3*4435200 = 0
    
    # q(x) = g(x)/x^2, so in terms of u = x^2:
    # Q(u) = a5*u^5 + a4*u^4 + a3*u^3 + a2*u^2 + a1*u + a0
    # where:
    # a5 = c3 * 4096
    # a4 = c3 * (-98304)
    # a3 = c2*256 + c3*1032192
    # a2 = c2*(-3072) + c3*(-5308416)
    # a1 = c1*16 + c2*11520 + c3*12096000
    # a0 = c1*(-48) + c2*(-13440) + c3*(-12902400)
    
    def compute_Q_coefficients(c0, c1, c2):
        c3 = -(c0 + 12*c1 + 1680*c2) / 4435200.0
        a5 = c3 * 4096
        a4 = c3 * (-98304)
        a3 = c2*256 + c3*1032192
        a2 = c2*(-3072) + c3*(-5308416)
        a1 = c1*16 + c2*11520 + c3*12096000
        a0 = c1*(-48) + c2*(-13440) + c3*(-12902400)
        return np.array([a5, a4, a3, a2, a1, a0])
    
    def objective(coeffs):
        c0, c1, c2 = coeffs
        Q_coeffs = compute_Q_coefficients(c0, c1, c2)
        # Q(u) = Q_coeffs[0]*u^5 + Q_coeffs[1]*u^4 + ... + Q_coeffs[5]
        # Find positive real roots
        roots = np.roots(Q_coeffs)
        # Filter for real positive roots
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        positive_roots = real_roots[real_roots > 1e-10]
        if len(positive_roots) == 0:
            return 0.0
        # We need roots where Q changes sign
        # For a polynomial, sign changes at simple roots
        # The largest positive root where Q changes sign
        # Let's check each positive root
        max_r2 = 0.0
        for r in positive_roots:
            # Check if Q changes sign at r
            eps = 1e-8
            Q_left = np.polyval(Q_coeffs, r - eps)
            Q_right = np.polyval(Q_coeffs, r + eps)
            if Q_left * Q_right < 0:
                if r*r > max_r2:
                    max_r2 = r*r
        return max_r2
    
    # Use the seed as initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Try multiple starting points
    best_result = None
    best_value = np.inf
    
    # Try seed and some perturbations
    starting_points = [x0.copy()]
    # Add some random-ish starting points
    np.random.seed(42)
    for _ in range(20):
        perturbation = np.random.uniform(-0.5, 0.5, 3)
        starting_points.append(x0 + perturbation)
    
    for start in starting_points:
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    else:
        return x0

# EVOLVE-BLOCK-END
