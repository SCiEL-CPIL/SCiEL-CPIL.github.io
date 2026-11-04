"""Refined uncertainty inequality: Laguerre polynomial zeros for Cohn-Goncalves construction."""
import numpy as np
from scipy.special import roots_laguerre
import math

M = 12
N = 1

def _laguerre_zeros_alpha_beta(alpha, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x).
    """
    if alpha == 0:
        x, w = roots_laguerre(degree)
        return np.sort(x)
    else:
        # Use Golub-Welsch for general alpha
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
    The score is the geometric mean of (1 + r_i) for the roots.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    return np.exp(np.mean(np.log(1 + roots)))

def _compute_laguerre_zeros(n, alpha=0):
    """Compute the n zeros of L_n^(alpha)(x)."""
    return _laguerre_zeros_alpha_beta(alpha, n)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the zeros of the Laguerre polynomial L_12^(0)(x) which minimize the
    geometric mean of (1+r_i) subject to the Laguerre orthogonality constraint.
    """
    # The optimal roots for the Cohn-Goncalves construction with m=12, n=1
    # are the zeros of the Laguerre polynomial L_12^(0)(x)
    roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        # Fallback: use scaled zeros
        roots = 4.0 * np.arange(1, 13, dtype=float)
    
    return [float(r) for r in roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
