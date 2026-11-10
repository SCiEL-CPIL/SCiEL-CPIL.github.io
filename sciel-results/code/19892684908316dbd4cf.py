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

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    # Grid coordinates
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Mean
    mu_x = 4.5
    mu_y = 4.5
    
    # Covariance parameters
    # We want strong negative correlation and sigma_x approx 2 * sigma_y
    sigma_x = 1.5
    sigma_y = 0.75
    rho = -0.9
    
    # Covariance matrix
    cov = np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])
    
    # Inverse covariance
    inv_cov = np.linalg.inv(cov)
    
    # Calculate Mahalanobis distance squared for each point
    # d^2 = [x-mu, y-mu] @ inv_cov @ [x-mu, y-mu]^T
    dx = X - mu_x
    dy = Y - mu_y
    
    # Vectorize calculation
    # z = [dx, dy]
    # d2 = z @ inv_cov @ z.T
    # Using einsum or explicit matrix ops
    z = np.stack([dx, dy], axis=-1)
    d2 = np.einsum('...i,ij,...j->...', z, inv_cov, z)
    
    # Gaussian PDF (unnormalized)
    # exp(-0.5 * d2)
    P = np.exp(-0.5 * d2)
    
    # Clip to grid (already done by meshgrid, but ensure non-negative)
    P[P < 0] = 0
    
    # Normalize
    total = P.sum()
    if total <= 1e-9:
        # Fallback to uniform if something went wrong
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total
        
    return P


# EVOLVE-BLOCK-END
