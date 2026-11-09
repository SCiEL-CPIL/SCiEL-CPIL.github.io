"""Symbolic regression with 2 inputs: Separable quadratic basis with orthonormalized Legendre polynomials."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _legendre_basis(t: np.ndarray) -> np.ndarray:
    """
    Compute orthonormalized Legendre polynomials of degree 0-3 on [-1, 1].
    
    Standard Legendre polynomials:
    P0(t) = 1
    P1(t) = t
    P2(t) = (3t^2 - 1)/2
    P3(t) = (5t^3 - 3t)/2
    
    Orthonormalization: multiply by sqrt((2n+1)/2) for n=0,1,2,3
    to make them orthonormal with respect to the L2 inner product on [-1,1].
    """
    n = t.shape[0]
    t2 = t * t
    t3 = t2 * t
    
    # Standard Legendre polynomials
    P0 = np.ones(n)
    P1 = t
    P2 = (3.0 * t2 - 1.0) / 2.0
    P3 = (5.0 * t3 - 3.0 * t) / 2.0
    
    # Orthonormalization factors: sqrt((2n+1)/2)
    c0 = np.sqrt(1.0 / 2.0)  # sqrt(1/2)
    c1 = np.sqrt(3.0 / 2.0)  # sqrt(3/2)
    c2 = np.sqrt(5.0 / 2.0)  # sqrt(5/2)
    c3 = np.sqrt(7.0 / 2.0)  # sqrt(7/2)
    
    L0 = c0 * P0
    L1 = c1 * P1
    L2 = c2 * P2
    L3 = c3 * P3
    
    return np.column_stack([L0, L1, L2, L3])


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix using a separable quadratic basis.
    
    The basis is f(x,y) ≈ sum_{i=0}^{3} sum_{j=0}^{2} a_i(x) * b_j(y)
    where a_i and b_j are orthonormalized Legendre polynomials.
    
    This gives 4 * 3 = 12 terms, but we select 10 to match the parameter count:
    - All 4 terms for j=0 (a0*b0, a1*b0, a2*b0, a3*b0)
    - All 3 terms for j=1 (a0*b1, a1*b1, a2*b1)
    - 3 terms for j=2 (a0*b2, a1*b2, a2*b2)
    
    This captures low-rank structure and avoids ill-conditioning from full tensor products.
    """
    # Compute orthonormal Legendre bases for x and y
    X_basis = _legendre_basis(x)  # (n, 4)
    Y_basis = _legendre_basis(y)  # (n, 4)
    
    # Extract columns
    a0, a1, a2, a3 = X_basis[:, 0], X_basis[:, 1], X_basis[:, 2], X_basis[:, 3]
    b0, b1, b2, b3 = Y_basis[:, 0], Y_basis[:, 1], Y_basis[:, 2], Y_basis[:, 3]
    
    # Separable features: a_i(x) * b_j(y)
    # 10 features selected for good coverage
    f0 = a0 * b0  # L0(x)*L0(y) - constant
    f1 = a1 * b0  # L1(x)*L0(y) - linear in x
    f2 = a0 * b1  # L0(x)*L1(y) - linear in y
    f3 = a2 * b0  # L2(x)*L0(y) - quadratic in x
    f4 = a0 * b2  # L0(x)*L2(y) - quadratic in y
    f5 = a1 * b1  # L1(x)*L1(y) - mixed linear
    f6 = a3 * b0  # L3(x)*L0(y) - cubic in x
    f7 = a0 * b3  # L0(x)*L3(y) - cubic in y
    f8 = a2 * b1  # L2(x)*L1(y) - quadratic-linear mixed
    f9 = a1 * b2  # L1(x)*L2(y) - linear-quadratic mixed
    
    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
