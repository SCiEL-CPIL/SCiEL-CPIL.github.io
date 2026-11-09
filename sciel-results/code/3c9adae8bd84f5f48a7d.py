import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combined approach: Chebyshev polynomial basis for conditioning
    + RBF (Gaussian kernel) terms for local flexibility.
    
    10 parameters:
    - params[0]: constant (DC offset)
    - params[1-3]: linear Chebyshev terms in x, y, z
    - params[4-6]: quadratic Chebyshev terms in x, y, z
    - params[7]: RBF weight (shared across all centers)
    - params[8]: RBF width (inverse sigma squared, positive)
    - params[9]: linear interaction term (Cheb1(x)*Cheb1(y)*Cheb1(z))
    
    The RBF centers are fixed at strategic points in [0.5, 2]^3.
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Map inputs to [-1, 1] for Chebyshev evaluation
    # Domain is [0.5, 2], center=1.25, half_range=0.75
    xc = (x - 1.25) / 0.75
    yc = (y - 1.25) / 0.75
    zc = (z - 1.25) / 0.75
    
    # Chebyshev polynomials
    T1x = xc
    T1y = yc
    T1z = zc
    T2x = 2.0 * xc * xc - 1.0
    T2y = 2.0 * yc * yc - 1.0
    T2z = 2.0 * zc * zc - 1.0
    
    # Trilinear interaction in Chebyshev space
    Txyz = T1x * T1y * T1z
    
    # RBF component: sum of Gaussian kernels at fixed centers
    # Use a small set of well-distributed centers
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.25, 1.25, 0.5]
    ])
    
    # Compute squared distances from each point to each center
    # For efficiency, compute in a vectorized manner
    # x, y, z are 1D arrays of length N
    # centers is (10, 3)
    
    # dx = x[:, None] - centers[:, 0][None, :]  -> shape (N, 10)
    # dy = y[:, None] - centers[:, 1][None, :]  -> shape (N, 10)
    # dz = z[:, None] - centers[:, 2][None, :]  -> shape (N, 10)
    
    dx = x[:, np.newaxis] - centers[np.newaxis, :, 0]
    dy = y[:, np.newaxis] - centers[np.newaxis, :, 1]
    dz = z[:, np.newaxis] - centers[np.newaxis, :, 2]
    
    sq_dist = dx * dx + dy * dy + dz * dz  # shape (N, 10)
    
    # RBF width: use exp(-width * sq_dist) where width = max(params[8], 0.1)
    # Clamp width to avoid numerical issues
    width = max(float(p[8]), 0.1)
    weight = float(p[7])
    
    # rbf_sum = sum of exp(-width * sq_dist) over centers
    # Clip sq_dist to avoid overflow in exp
    sq_dist_clipped = np.minimum(sq_dist, 50.0 / max(width, 0.1))
    rbf_values = np.exp(-width * sq_dist_clipped)  # shape (N, 10)
    rbf_sum = np.sum(rbf_values, axis=1)  # shape (N,)
    
    # Combine all terms
    result = (p[0] * 1.0
              + p[1] * T1x
              + p[2] * T1y
              + p[3] * T1z
              + p[4] * T2x
              + p[5] * T2y
              + p[6] * T2z
              + p[9] * Txyz
              + weight * rbf_sum)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
