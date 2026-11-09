"""Symbolic regression with separable quadratic Legendre basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _legendre_basis(t: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute orthonormalized Legendre polynomials of degree 0..degree
    evaluated at t (assumed in [-1,1]).
    Returns array of shape (len(t), degree+1).
    """
    n = len(t)
    d = degree + 1
    basis = np.zeros((n, d))
    
    # P_0 = 1
    basis[:, 0] = 1.0
    if degree >= 1:
        # P_1 = t
        basis[:, 1] = t
    for k in range(2, d):
        # P_k = ((2k-1)*t*P_{k-1} - (k-1)*P_{k-2}) / k
        basis[:, k] = ((2*k - 1) * t * basis[:, k-1] - (k-1) * basis[:, k-2]) / k
    
    # Orthonormalize: each Legendre polynomial P_k on [-1,1] has norm sqrt(2/(2k+1))
    # So orthonormal basis is P_k * sqrt((2k+1)/2)
    for k in range(d):
        norm = np.sqrt(2.0 / (2*k + 1))
        basis[:, k] /= norm
    
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable quadratic basis: f(x,y) ≈ sum_{i=0}^{3} sum_{j=0}^{3} c_{ij} * L_i(x) * L_j(y)
    where L_i are orthonormal Legendre polynomials.
    
    params has length 10. We use a low-rank + diagonal structure to fit 10 params:
    - 4 params for the diagonal of the 4x4 coefficient matrix: c_00, c_11, c_22, c_33
    - 3 params for off-diagonal symmetric pairs: c_01, c_12, c_23 (and their symmetric counterparts)
    - 3 params for marginal corrections: x-only, y-only, and constant interaction
    
    Actually, let's use a cleaner structure with exactly 10 parameters:
    We'll use a rank-1 + diagonal + low-order cross structure.
    
    Let's define:
    f(x,y) = a0*L0(x)*L0(y) + a1*L1(x)*L1(y) + a2*L2(x)*L2(y) + a3*L3(x)*L3(y)
           + b0*L0(x)*L1(y) + b1*L1(x)*L0(y) + b2*L2(x)*L1(y) + b3*L1(x)*L2(y)
           + c0*L0(x)*L2(y) + c1*L2(x)*L0(y)
    
    This gives 10 parameters total.
    """
    # Compute orthonormal Legendre basis for x and y up to degree 3
    Lx = _legendre_basis(x, 3)  # shape (n, 4)
    Ly = _legendre_basis(y, 3)  # shape (n, 4)
    
    # Extract components
    Lx0, Lx1, Lx2, Lx3 = Lx[:, 0], Lx[:, 1], Lx[:, 2], Lx[:, 3]
    Ly0, Ly1, Ly2, Ly3 = Ly[:, 0], Ly[:, 1], Ly[:, 2], Ly[:, 3]
    
    # 10 parameters:
    # Diagonal terms: c00, c11, c22, c33
    # Off-diagonal: c01 (Lx0*Ly1), c10 (Lx1*Ly0), c21 (Lx2*Ly1), c12 (Lx1*Ly2), c02 (Lx0*Ly2), c20 (Lx2*Ly0)
    # That's 4 + 6 = 10 terms
    
    p = params
    
    # Diagonal
    term_diag = (p[0] * Lx0 * Ly0 +
                 p[1] * Lx1 * Ly1 +
                 p[2] * Lx2 * Ly2 +
                 p[3] * Lx3 * Ly3)
    
    # Off-diagonal (6 terms)
    term_off = (p[4] * Lx0 * Ly1 +   # c01
                p[5] * Lx1 * Ly0 +   # c10
                p[6] * Lx2 * Ly1 +   # c21
                p[7] * Lx1 * Ly2 +   # c12
                p[8] * Lx0 * Ly2 +   # c02
                p[9] * Lx2 * Ly0)    # c20
    
    result = term_diag + term_off
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
