# EVOLVE-BLOCK-START
import numpy as np


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    This implementation uses a distribution that weights cells by (1 + |x - y|) to flatten
    the distribution of x - y, thereby maximizing H_{-1}, while accepting a potential
    increase in the marginal entropies.
    """
    n = 10
    # Create coordinate grids
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy') # X is x, Y is y
    
    # Calculate weights
    # P[y, x] corresponds to X[y, x] and Y[y, x]
    # Note: meshgrid with indexing='xy' gives X[i, j] = j, Y[i, j] = i
    # So X is the column index (x), Y is the row index (y).
    # We want P[y, x] = weight(y, x)
    weights = 1.0 + np.abs(X - Y)
    
    # Normalize
    total_weight = np.sum(weights)
    P = weights / total_weight
    
    return P


# EVOLVE-BLOCK-END
