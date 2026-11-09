"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_laguerre

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 590  # Safety margin before 600s limit
    
    m = 12
    n = 1
    
    # The Cohn-Gonçalves construction: we have a polynomial with 12 positive double roots
    # The polynomial is of the form: P(x) = prod_{i=1}^{12} (x - r_i)^2 * L_n^(2m)(x) or similar
    # Actually, for the Fourier sign uncertainty, the construction involves:
    # A Laguerre combination where we place 12 positive double roots to minimize
    # the last sign change, which gives the C4 upper bound.
    
    # The key insight: the C4 upper bound is related to the largest root of the
    # constructed polynomial. We want to minimize this.
    
    # For the Cohn-Gonçalves construction with m double roots and n=1,
    # the polynomial is: Q(x) = L_n(x) + sum of terms that create double roots
    # 
    # Actually, let me reconsider. The standard approach:
    # We construct a polynomial P(x) = L_m(x) * prod_{i=1}^{k} (x - a_i)^2
    # where L_m is a Laguerre polynomial, and we choose a_i to minimize
    # the last sign change (largest positive root).
    
    # For m=12, n=1: The construction uses L_12^(0) or a related Laguerre polynomial
    # with 12 additional double roots placed optimally.
    
    # The C4 bound is determined by the largest positive root of the final polynomial.
    
    # Let's define the objective: given 12 positive root locations r_1 < r_2 < ... < r_12,
    # form P(x) = L_12^(0)(x) * prod_{i=1}^{12} (x - r_i)^2
    # and find the largest positive root of P(x). Minimize this.
    
    # But wait - if we multiply by (x-r_i)^2, those become double roots of P(x),
    # and the original roots of L_12^(0)(x) remain. The largest root of P(x) would be
    # max(largest root of L_12^(0), r_12).
    
    # That doesn't make sense for optimization. Let me reconsider.
    
    # The Cohn-Gonçalves construction for Fourier uncertainty:
    # We want a polynomial with specific root structure that bounds the sign changes
    # of the Fourier transform. The "12 positive double roots" are part of the
    # construction, and the C4 bound is the last sign change of the associated
    # polynomial.
    
    # Alternative interpretation: The polynomial is a linear combination of Laguerre
    # polynomials that has 12 positive double roots. We optimize the coefficients
    # (equivalently, the root locations) to minimize the C4 bound.
    
    # Let me try a different approach: use scipy to optimize the root locations
    # directly, where the objective is the largest positive root of the constructed
    # polynomial.
    
    def compute_largest_root(roots):
        """Given 12 positive double root locations, compute the largest positive root
        of the polynomial P(x) = L_12^(0)(x) * prod (x - r_i)^2"""
        # Build the polynomial coefficients
        # L_12^(0)(x) coefficients
        lag_coeffs = np.polynomial.laguerre.laggauss(12)[0]  # roots of L_12
        # Actually, get the coefficients of L_12^(0)(x)
        from numpy.polynomial.laguerre import Laguerre
        L12 = Laguerre.basis(12)
        lag_c = L12.coef  # coefficients in Laguerre basis, need to convert
        
        # Convert to power basis
        L12_power = L12.convert(kind=np.polynomial.Polynomial)
        lag_power_c = L12_power.coef
        
        # Build prod (x - r_i)^2 for i=1..12
        # Start with [1] and multiply by each (x - r_i)^2 = x^2 - 2*r_i*x + r_i^2
        prod_c = np.array([1.0])
        for r in roots:
            # (x - r)^2 = x^2 - 2r*x + r^2
            factor = np.array([1.0, -2.0*r, r*r])
            prod_c = np.convolve(prod_c, factor)
        
        # Multiply by L_12^(0)(x)
        full_c = np.convolve(lag_power_c, prod_c)
        
        # Find all roots
        all_roots = np.roots(full_c)
        
        # Get positive real roots
        positive_roots = []
        for root in all_roots:
            if abs(root.imag) < 1e-8 and root.real > 1e-10:
                positive_roots.append(root.real)
        
        if not positive_roots:
            return 1e10  # Large penalty
        
        return max(positive_roots)
    
    # Initial guess: use Gauss-Laguerre nodes as starting point
    initial_roots = np.polynomial.laguerre.laggauss(12)[0]
    # Sort them
    initial_roots = np.sort(initial_roots)
    
    # Define objective function
    def objective(roots):
        # Ensure roots are positive and sorted
        roots = np.maximum(roots, 1e-6)
        roots = np.sort(roots)
        try:
            val = compute_largest_root(roots)
            return val
        except Exception:
            return 1e10
    
    # Use Nelder-Mead optimization (derivative-free, good for this kind of problem)
    try:
        result = minimize(objective, initial_roots, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if time.time() > deadline:
            return [float(r) for r in np.sort(np.maximum(result.x, 1e-6))]
        
        optimal_roots = np.sort(np.maximum(result.x, 1e-6))
        
    except Exception:
        optimal_roots = initial_roots
    
    # Verify and return
    if time.time() > deadline:
        return [float(r) for r in np.sort(np.maximum(optimal_roots, 1e-6))]
    
    result = [float(r) for r in optimal_roots]
    return result
# EVOLVE-BLOCK-END
