"""Improved program: Optimize roots for Laguerre double-root construction."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize

M = 12          # number of double roots
N = 1           # Laguerre order (n=1)
K = 2 * M + 1   # number of basis Laguerres (degree up to 2M)

def _get_laguerre_coefficients_fast(roots):
    """
    Fast algebraic computation of Laguerre coefficients.
    P(x) = prod (x - r_i)^2 = sum_{j=0}^{2M} a_j x^j
    c_k = 1/h_k * integral_0^inf x^N e^{-x} P(x) L_k^{(N)}(x) dx
    Using expansion of x^j in Laguerre basis or direct integral evaluation.
    """
    # 1. Get monomial coefficients of P(x)
    # np.poly(roots) gives coeffs of prod(x-r_i). 
    # We want prod(x-r_i)^2.
    # We can just square the polynomial? No, (sum a_i x^i)^2 != sum a_i^2 x^{2i}.
    # Easiest: pass each root twice to np.poly.
    double_roots = np.repeat(roots, 2)
    # np.poly returns coefficients in descending powers: c_{2M} x^{2M} + ... + c_0
    poly_coeffs = np.poly(double_roots)
    
    # a_j is coefficient of x^j.
    # poly_coeffs[0] is coeff of x^{2M}.
    # So a_j = poly_coeffs[2M - j].
    a = np.zeros(2 * M + 1)
    for j in range(2 * M + 1):
        a[j] = poly_coeffs[2 * M - j]
        
    coeffs = np.zeros(K)
    
    # Precompute factorials/gamma terms for efficiency
    # h_k = Gamma(k+N+1)/k! = Gamma(k+2)/k! = k+1 for N=1
    # Integral term: I_{j,k} = sum_{m=0}^k binom(k+1, k-m) * (-1)^m/m! * Gamma(j+2+m)
    
    # To avoid recomputing Gamma, we can use log-gamma or precompute a table.
    # Max j is 2M=24. Max m is K-1=24. Max arg to Gamma is 24+2+24=50.
    # This is small.
    
    for k in range(K):
        h_k = k + 1.0 # For N=1
        integral_val = 0.0
        for m in range(k + 1):
            # binom(k+1, k-m) = binom(k+1, m)
            binom_val = 1.0
            # Compute binom(k+1, m)
            # Using math.comb is available in 3.8+
            import math
            binom_val = math.comb(k + 1, m)
            
            # (-1)^m / m!
            sign = 1.0 if m % 2 == 0 else -1.0
            fact_m = math.factorial(m)
            term_coeff = binom_val * sign / fact_m
            
            # Sum over j
            # I_{j,k} = term_coeff * Gamma(j+2+m)
            # We need sum_{j=0}^{2M} a_j * term_coeff * Gamma(j+2+m)
            # = term_coeff * sum_{j=0}^{2M} a_j * Gamma(j+2+m)
            
            # We can precompute S_m = sum_{j=0}^{2M} a_j * Gamma(j+2+m) for each m?
            # Or just loop j. 25 * 25 operations is trivial.
            s_val = 0.0
            for j in range(2 * M + 1):
                if a[j] == 0: continue
                s_val += a[j] * math.exp(gammaln(j + 2 + m))
            
            integral_val += term_coeff * s_val
            
        coeffs[k] = integral_val / h_k
        
    return coeffs

def _validate_roots(roots):
    """Check if roots are valid and coefficients are non-negative."""
    if len(roots) != M:
        return False
    if np.any(roots <= 0):
        return False
    if np.any(np.diff(roots) <= 1e-6):
        return False
    
    coeffs = _get_laguerre_coefficients_fast(roots)
    if coeffs is None:
        return False
    
    # Check non-negativity of coefficients
    # Allow small numerical tolerance
    if np.any(coeffs < -1e-8):
        return False
        
    return True

def construct():
    # Initial guess: roots of L_M^{(N)}(x)
    try:
        roots0, _ = roots_genlaguerre(M, N)
        roots0 = np.sort(roots0)
    except Exception:
        roots0 = np.array([i * 4.0 for i in range(1, M + 1)])

    def objective(params):
        roots = np.sort(params)
        # Penalty for invalid roots
        if np.any(roots <= 0):
            return 1e10
        if np.any(np.diff(roots) <= 1e-6):
            return 1e10
            
        coeffs = _get_laguerre_coefficients_fast(roots)
        if coeffs is None:
            return 1e10
            
        # Penalty for negative coefficients
        neg_min = np.min(coeffs)
        if neg_min < 0:
            # Large penalty proportional to how negative it is
            return 1e6 * (-neg_min) + np.max(roots)
            
        return np.max(roots)

    # Use multiple random restarts with L-BFGS-B
    best_roots = roots0
    best_score = objective(roots0)
    
    np.random.seed(42)
    # Try several perturbations
    for i in range(20):
        # Perturb roots
        scale = 0.1 * (1 - i/20.0) # Decreasing perturbation
        perturbed = roots0 * (1 + scale * np.random.randn(M))
        perturbed = np.sort(perturbed)
        if np.any(perturbed <= 0):
            continue
            
        try:
            res = minimize(objective, perturbed, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x
        except Exception:
            pass

    # Final validation
    if not _validate_roots(best_roots):
        # Fallback to initial Laguerre roots if optimization broke validity
        if _validate_roots(roots0):
            best_roots = roots0
        else:
            # Ultimate fallback: uniform
            best_roots = np.array([i * 4.0 for i in range(1, M + 1)])

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
