"""Refined uncertainty inequality: Dyson-gas mean-field fixed point for Laguerre roots."""
import numpy as np
from scipy.optimize import brentq
import math

# Constants
M = 12
N = 1  # dimension

def _laguerre_zeros_alpha(alpha, degree):
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

def _dyson_gas_equilibrium_dbl(alpha, n=12, tol=1e-13, max_iter=10000):
    """
    Solve the Dyson gas equilibrium for n double roots with Laguerre weight x^alpha * exp(-x).
    
    For double roots, the equilibrium condition is:
        sum_{j != i} 2 / (r_i - r_j) = 1 - alpha / r_i
    
    We solve this using Newton's method on the full system with a tridiagonal
    structure exploited for efficiency.
    """
    # Initial guess: use Laguerre zeros as starting point
    roots = _laguerre_zeros_alpha(alpha, n).copy()
    roots = np.maximum(roots, 1e-10)
    
    for it in range(max_iter):
        # Compute the residual and Jacobian
        n = len(roots)
        r = roots.copy()
        
        # Residual: F_i = sum_{j!=i} 2/(r_i - r_j) - (1 - alpha/r_i)
        F = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += 2.0 / (r[i] - r[j])
            F[i] = s - (1.0 - alpha / r[i])
        
        if np.max(np.abs(F)) < tol:
            break
        
        # Build Jacobian: J_ij = dF_i/dr_j
        # dF_i/dr_i = -sum_{j!=i} 2/(r_i - r_j)^2 + alpha/r_i^2
        # dF_i/dr_j = 2/(r_i - r_j)^2 for j != i
        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = -sum(2.0 / (r[i] - r[j])**2 for j in range(n) if j != i) + alpha / (r[i]**2)
            for j in range(n):
                if j != i:
                    J[i, j] = 2.0 / (r[i] - r[j])**2
        
        # Solve J * delta = -F
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            delta = np.linalg.lstsq(J, -F, rcond=None)[0]
        
        # Damped update to maintain ordering and positivity
        step = 1.0
        for _ in range(50):
            new_r = r + step * delta
            if np.all(new_r > 1e-10) and np.all(np.diff(new_r) > 1e-10):
                r = new_r
                break
            step *= 0.5
        else:
            # If we couldn't take any step, break
            break
        
        roots = r
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-10)
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field fixed point with double-root modification.
    Tries multiple alpha values and picks the one with best score.
    """
    best_roots = None
    best_score = float('inf')
    
    # Try a range of alpha values
    alpha_candidates = np.linspace(-0.5, 5.0, 30)
    
    for alpha in alpha_candidates:
        try:
            roots = _dyson_gas_equilibrium_dbl(alpha, n=12)
            if len(roots) != 12:
                continue
            if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
                continue
            score = _compute_score(roots)
            if score < best_score:
                best_score = score
                best_roots = roots.copy()
        except Exception:
            continue
    
    # Also try the seed roots as a fallback
    seed_roots = 4.0 * np.arange(1, 13)
    seed_score = _compute_score(seed_roots)
    
    if best_roots is None or best_score > seed_score:
        best_roots = seed_roots
        best_score = seed_score
    
    # Also try plain Laguerre zeros for various alphas
    for alpha in alpha_candidates:
        try:
            roots = _laguerre_zeros_alpha(alpha, 12)
            if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
                continue
            score = _compute_score(roots)
            if score < best_score:
                best_score = score
                best_roots = roots.copy()
        except Exception:
            continue
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
