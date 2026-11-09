"""Rewritten symbolic regression candidate: separable quadratic basis with orthonormal Legendre polynomials."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _legendre_orthonormal(t, degree):
    """Return orthonormal Legendre polynomial values on [-1,1] up to given degree.
    Uses the three-term recurrence with normalization for orthonormality.
    L_n^orth(t) = sqrt((2n+1)/2) * P_n(t) where P_n is standard Legendre.
    """
    n = int(degree)
    # P_0 = 1, P_1 = t
    P0 = np.ones_like(t, dtype=float)
    P1 = t.copy()
    # Normalize: sqrt((2*0+1)/2)*P0 = sqrt(1/2), sqrt((2*1+1)/2)*P1 = sqrt(3/2)*t
    L0 = np.sqrt(0.5) * P0
    L1 = np.sqrt(1.5) * P1
    polys = [L0, L1]
    for k in range(2, n + 1):
        # (k) P_k = (2k-1) t P_{k-1} - (k-1) P_{k-2}
        Pk = ((2 * k - 1) * t * P1 - (k - 1) * P0) / k
        Lk = np.sqrt((2 * k + 1) / 2.0) * Pk
        polys.append(Lk)
        P0, P1 = P1, Pk
    return np.stack(polys, axis=1)  # shape (N, degree+1)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable quadratic basis: f(x,y) ≈ sum_{i,j} a_i(x) b_j(y) * c_{ij}
    where a_i, b_j are orthonormal Legendre polynomials of degree 0..3.
    That's 4x4 = 16 coefficients, but we only have 10 params.
    
    Strategy: Use a low-rank separable form with 10 params:
    f(x,y) = sum_{k=0}^{3} alpha_k * L_k(x) * L_k(y)   (4 diagonal terms)
            + sum_{k=0}^{2} beta_k * L_k(x)             (3 x-only terms)
            + sum_{k=0}^{2} gamma_k * L_k(y)            (3 y-only terms)
    Total: 4 + 3 + 3 = 10 params.
    
    This captures:
    - Diagonal interactions (x-y coupling through shared degree)
    - Independent x and y effects
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Orthonormal Legendre polynomials up to degree 3
    Lx = _legendre_orthonormal(x, 3)  # shape (N, 4): columns are L0, L1, L2, L3
    Ly = _legendre_orthonormal(y, 3)  # shape (N, 4)
    
    # Unpack 10 params
    # params[0:4] = alpha_0..alpha_3 (diagonal interaction terms)
    # params[4:7] = beta_0..beta_2 (x-only terms)
    # params[7:10] = gamma_0..gamma_2 (y-only terms)
    alpha = params[0:4]
    beta = params[4:7]
    gamma = params[7:10]
    
    # Diagonal interaction: sum_k alpha_k * L_k(x) * L_k(y)
    interaction = np.sum(alpha[None, :] * Lx * Ly, axis=1)
    
    # X-only terms: sum_{k=0}^{2} beta_k * L_k(x)
    x_term = np.sum(beta[None, :] * Lx[:, :3], axis=1)
    
    # Y-only terms: sum_{k=0}^{2} gamma_k * L_k(y)
    y_term = np.sum(gamma[None, :] * Ly[:, :3], axis=1)
    
    return interaction + x_term + y_term
# EVOLVE-BLOCK-END
