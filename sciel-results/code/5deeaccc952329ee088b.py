"""Refined uncertainty inequality: Dyson-gas equilibrium with linear confining potential."""
import numpy as np
from scipy.optimize import brentq
import math

# Constants
M = 12
N = 1  # dimension

def _laguerre_zeros_alpha_beta(alpha, degree):
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

def _dyson_gas_with_linear_confinement(n_roots, beta=2.0, gamma=1.0):
    """
    Solve the Dyson gas equilibrium with repulsion exponent beta=2 (double roots)
    and linear confining potential V(x) = gamma * x.
    
    The equilibrium condition for each root r_i is:
        sum_{j != i} beta / (r_i - r_j) = gamma / 2 + 1/(2*r_i)
    
    This comes from the log-derivative of the weight function w(x) = x * exp(-gamma*x)
    for a Dyson gas with double-root repulsion.
    """
    n = n_roots
    # Initial guess: spread out linearly
    roots = np.linspace(0.1, n * 0.5, n)
    
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
                right = other_sorted[0] - 1e-8
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-8
                right = max(50.0, r_guess + 10.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-8
                right = other_sorted[insert_pos] - 1e-8
            
            if left < 1e-10:
                left = 1e-10
            if right <= left:
                left = max(1e-10, r_guess - 2.0)
                right = r_guess + 2.0
            
            def f(r):
                if r <= 1e-12:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-12):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return beta * s - gamma / 2.0 - 1.0 / (2.0 * r)
            
            found = False
            for attempt in range(100):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if not np.isfinite(f_left) or not np.isfinite(f_right):
                        left = max(1e-10, left - 0.5)
                        right = right + 0.5
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
                        if f_left > 0 and f_right > 0:
                            # Both positive, need to move right
                            right = right + 1.0
                        elif f_left < 0 and f_right < 0:
                            # Both negative, need to move left
                            left = max(1e-10, left - 1.0)
                        else:
                            left = max(1e-10, left - 1.0)
                            right = right + 1.0
                except Exception:
                    left = max(1e-10, left - 1.0)
                    right = right + 1.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-15:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-10)
    
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas equilibrium with repulsion exponent 2 and linear confining potential.
    """
    # Solve Dyson gas with double-root repulsion (beta=2) and linear confinement
    # The linear confinement corresponds to the Laguerre weight e^{-x}
    roots = _dyson_gas_with_linear_confinement(12, beta=2.0, gamma=1.0)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        # Fallback to Laguerre zeros
        roots = _laguerre_zeros_alpha_beta(0, 12)
    
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
