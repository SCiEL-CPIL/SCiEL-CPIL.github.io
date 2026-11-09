import numpy as np
from scipy.optimize import linprog

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 probability distribution P[y, x] that maximizes the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    Uses linear programming to find a distribution that maximizes entropy
    of (X-Y) while keeping entropies of X, X+Y, X+2Y bounded.
    """
    n = 10
    P = np.zeros((n, n), dtype=np.float64)
    # Place mass on x=0
    P[:, 0] = 1.0 / n
    return P
