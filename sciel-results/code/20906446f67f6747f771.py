"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_basis(z: np.ndarray, degree: int) -> np.ndarray:
    """Compute Chebyshev polynomials T_0(z), ..., T_degree(z) via recurrence.
    
    Returns array of shape (N, degree+1).
    """
    N = z.shape[0]
    basis = np.empty((N, degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = z
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * z * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Chebyshev model:
        f(x) = (sum_{i=0}^{5} a_i * T_i(z)) / (sum_{j=0}^{3} b_j * T_j(z) + eps)
    where z = x / 2 maps [-2, 2] -> [-1, 1].
    
    params[0:6] are numerator coefficients a_0..a_5
    params[6:10] are denominator coefficients b_0..b_3
    """
    z = x / 2.0
    
    # Numerator: degree 5 Chebyshev basis (6 terms)
    num_basis = _chebyshev_basis(z, 5)
    # Denominator: degree 3 Chebyshev basis (4 terms)
    den_basis = _chebyshev_basis(z, 3)
    
    a = params[0:6]
    b = params[6:10]
    
    numerator = num_basis @ a
    denominator = den_basis @ b
    
    # Add small epsilon to avoid division by zero
    # The sign of epsilon should match the typical sign of denominator
    # Use a robust approach: add epsilon * sign(denominator) but that's non-smooth
    # Instead, just add a small constant; BFGS can handle this
    eps = 1e-8
    denominator = denominator + eps
    
    # Guard against very small denominators to prevent numerical issues
    # Clip denominator to avoid extreme values
    min_den = 1e-6
    mask = np.abs(denominator) < min_den
    if np.any(mask):
        # Preserve sign
        sign = np.sign(denominator[mask])
        sign[sign == 0] = 1.0
        denominator[mask] = sign * min_den
    
    result = numerator / denominator
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
