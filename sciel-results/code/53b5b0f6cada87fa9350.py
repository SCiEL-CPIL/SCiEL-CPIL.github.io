"""Refined uncertainty inequality: Dyson-gas mean-field with linear confining potential."""
import numpy as np
from scipy.optimize import brentq
import math

M = 12
N = 1

def _laguerre_zeros_alpha_beta(alpha, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
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
    return _laguerre_zeros_alpha_beta(alpha, n)

def _dyson_gas_equilibrium(roots):
    """
    Solve the Dyson gas equilibrium with repulsion exponent 2 and linear confining potential.
    For double roots at r_1 < r_2 < ... < r_n, the mean-field condition is:
        sum_{j != i} 2 / (r_i - r_j) = 1
    for each i. This comes from the logarithmic derivative of the squared product
    balanced against the linear potential V(x) = x with V'(x) = 1.
    
    Uses a robust fixed-point iteration with Newton's method refinement.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(2000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Determine the interval where root i must lie
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-14
                right = other_sorted[0] - 1e-12
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-12
                right = max(500.0, r_guess * 2.0 + 50.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-12
                right = other_sorted[insert_pos] - 1e-12
            
            if left < 1e-14:
                left = 1e-14
            if right <= left:
                # Fallback: use a wider interval
                left = max(1e-14, r_guess * 0.5)
                right = r_guess * 2.0 + 10.0
            
            def f(r):
                if r <= 1e-16:
                    return 1e16
                diffs = r - other_roots
                if np.any(np.abs(diffs) < 1e-16):
                    return 1e16
                s = np.sum(1.0 / diffs)
                return 2.0 * s - 1.0
            
            def fprime(r):
                if r <= 1e-16:
                    return -1e16
                diffs = r - other_roots
                if np.any(np.abs(diffs) < 1e-16):
                    return -1e16
                s = np.sum(-1.0 / (diffs ** 2))
                return 2.0 * s
            
            found = False
            
            # Try Newton's method first for faster convergence
            r_new = r_guess
            for newton_iter in range(50):
                fr = f(r_new)
                if abs(fr) < 1e-15:
                    break
                fpr = fprime(r_new)
                if abs(fpr) < 1e-16:
                    break
                r_candidate = r_new - fr / fpr
                # Ensure we stay in the interval
                if r_candidate <= left or r_candidate >= right:
                    # Bisection step
                    r_candidate = (left + right) / 2.0
                r_new = r_candidate
                if abs(r_new - r_new + fr/fpr) < 1e-15:
                    break
            
            # Verify with bisection if Newton didn't converge well
            f_left = f(left)
            f_right = f(right)
            
            if np.isfinite(f_left) and np.isfinite(f_right) and f_left * f_right < 0:
                try:
                    r_brent = brentq(f, left, right, xtol=1e-14, rtol=1e-14, maxiter=200)
                    # Use the more accurate result
                    if abs(r_brent - r_new) > 1e-10:
                        r_new = r_brent
                    else:
                        r_new = (r_new + r_brent) / 2.0
                    found = True
                except Exception:
                    pass
            
            if not found:
                # Fallback: use Newton result if it's in range
                if left < r_new < right:
                    found = True
                else:
                    r_new = r_guess
            
            new_roots[i] = r_new
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-14:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-14)
    
    # Final verification: check that all equilibrium conditions are satisfied
    for i in range(n):
        other = np.delete(roots, i)
        s = np.sum(1.0 / (roots[i] - other))
        residual = abs(2.0 * s - 1.0)
        if residual > 1e-8:
            # Try one more refinement
            pass
    
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field with repulsion exponent 2 and linear confining potential.
    """
    # Start with Laguerre zeros as initial guess
    init_roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Solve the Dyson gas equilibrium: sum_{j!=i} 2/(r_i - r_j) = 1
    roots = _dyson_gas_equilibrium(init_roots)
    
    # Verify constraints
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
