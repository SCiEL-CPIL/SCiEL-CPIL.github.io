"""Refined uncertainty inequality: Dyson-gas mean-field fixed point for Laguerre roots."""
import numpy as np
from scipy.optimize import brentq
import math

# Constants
M = 12
N = 1  # dimension

def _laguerre_zeros_alpha_beta(alpha, beta, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
    For Laguerre: weight x^alpha e^{-x}, alpha > -1.
    """
    n = degree
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2 * (i + 1) + alpha + 1
    for i in range(n - 1):
        val = (i + 1) * (i + 1 + alpha)
        J[i, i + 1] = math.sqrt(val)
        J[i + 1, i] = math.sqrt(val)
    
    eigvals = np.linalg.eigvalsh(J)
    return np.sort(eigvals)

def _compute_score(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    
    seed_roots = 4.0 * np.arange(1, 13)
    seed_score = np.exp(np.mean(np.log(1 + seed_roots)))
    current_score = np.exp(np.mean(np.log(1 + roots)))
    
    return current_score / seed_score

def _compute_laguerre_zeros(n, alpha=0):
    """Compute the n zeros of L_n^(alpha)(x)."""
    return _laguerre_zeros_alpha_beta(alpha, 0, n)

def _dyson_gas_equilibrium(roots, alpha=0.0):
    """
    Solve the Dyson gas equilibrium for double roots of a Laguerre-type polynomial.
    
    For P(x) = prod_{j=1}^{n} (x - r_j)^2 with Laguerre weight w(x) = x^alpha * exp(-x),
    the equilibrium condition at each double root r_i is:
    
        2 * sum_{j != i} 1/(r_i - r_j) = alpha / r_i - 1
    
    i.e., f(r_i) = 2 * sum_{j != i} 1/(r_i - r_j) - alpha/r_i + 1 = 0
    
    We solve this system via successive over-relaxation with brentq.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(1000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Find bracket between neighboring roots
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-10
                right = other_sorted[0] - 1e-10
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-10
                right = max(200.0, r_guess * 2.0 + 10.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-10
                right = other_sorted[insert_pos] - 1e-10
            
            if left < 1e-10:
                left = 1e-10
            if right <= left:
                left = max(1e-10, r_guess * 0.5)
                right = r_guess * 1.5 + 1.0
            
            def f(r):
                if r <= 1e-12:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-12):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return 2.0 * s - alpha / r + 1.0
            
            found = False
            for attempt in range(100):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if np.isnan(f_left) or np.isnan(f_right) or np.isinf(f_left) or np.isinf(f_right):
                        left = max(1e-10, left * 0.5)
                        right = right * 1.5 + 1.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-15, rtol=1e-15, maxiter=300)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-15:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-15:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        # Expand bracket
                        left = max(1e-10, left - 1.0)
                        right = right + 2.0
                except Exception:
                    left = max(1e-10, left - 1.0)
                    right = right + 2.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-14:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-10)
    
    # Ensure strict ordering
    for i in range(1, len(roots)):
        if roots[i] <= roots[i-1]:
            roots[i] = roots[i-1] + 1e-10
    
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field fixed point with double-root repulsion and Laguerre confining potential.
    """
    # Start with Laguerre zeros as initial guess
    init_roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Solve the Dyson gas equilibrium for double roots with alpha=0 (standard Laguerre weight)
    # Equilibrium: 2 * sum_{j!=i} 1/(r_i - r_j) = -1  (since alpha=0)
    roots = _dyson_gas_equilibrium(init_roots, alpha=0.0)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        # Fallback: try with a small alpha to get valid roots
        roots = _dyson_gas_equilibrium(init_roots, alpha=0.5)
        if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
            roots = _compute_laguerre_zeros(12, alpha=0)
    
    return [float(r) for r in roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
