"""
Symbolic regression for a nonlinear oscillator using a 2D Chebyshev polynomial basis.
This structural change replaces the sparse physics-informed monomial basis with a 
complete, orthogonal polynomial basis (Chebyshev) of total degree 3. 
This improves numerical conditioning and allows the optimizer to explore a more 
uniform feature space, potentially escaping local minima associated with 
ill-conditioned monomial bases or restrictive physics assumptions.
"""
import numpy as np
from scipy.optimize import minimize
import time

# Normalization constants derived from training ranges
X_MIN = -0.5728
X_MAX = 0.5275
V_MIN = -0.1754
V_MAX = 0.2821

# Map inputs to [-1, 1] for Chebyshev polynomials
X_SCALE = 2.0 / (X_MAX - X_MIN)
X_SHIFT = -(X_MAX + X_MIN) / (X_MAX - X_MIN)

V_SCALE = 2.0 / (V_MAX - V_MIN)
V_SHIFT = -(V_MAX + V_MIN) / (V_MAX - V_MIN)

def _compute_basis(x, v):
    """
    Compute the 10 Chebyshev basis functions.
    Returns a matrix of shape (N, 10).
    """
    X = x * X_SCALE + X_SHIFT
    V = v * V_SCALE + V_SHIFT
    
    # Compute Chebyshev polynomials for X
    T0_X = np.ones_like(X)
    T1_X = X
    T2_X = 2.0 * X**2 - 1.0
    T3_X = 4.0 * X**3 - 3.0 * X
    
    # Compute Chebyshev polynomials for V
    T0_V = np.ones_like(V)
    T1_V = V
    T2_V = 2.0 * V**2 - 1.0
    T3_V = 4.0 * V**3 - 3.0 * V
    
    # Construct the 10 basis functions
    b0 = T0_X * T0_V
    b1 = T1_X * T0_V
    b2 = T0_X * T1_V
    b3 = T2_X * T0_V
    b4 = T1_X * T1_V
    b5 = T0_X * T2_V
    b6 = T3_X * T0_V
    b7 = T2_X * T1_V
    b8 = T1_X * T2_V
    b9 = T0_X * T3_V
    
    return np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression using a 2D Chebyshev basis of total degree 3.
    
    Basis functions (where X, V are normalized to [-1, 1]):
    0: T0(X)T0(V) = 1
    1: T1(X)T0(V) = X
    2: T0(X)T1(V) = V
    3: T2(X)T0(V) = 2X^2 - 1
    4: T1(X)T1(V) = X*V
    5: T0(X)T2(V) = 2V^2 - 1
    6: T3(X)T0(V) = 4X^3 - 3X
    7: T2(X)T1(V) = (2X^2 - 1)*V
    8: T1(X)T2(V) = X*(2V^2 - 1)
    9: T0(X)T3(V) = 4V^3 - 3V
    
    This set spans the space of all polynomials in X and V with total degree <= 3.
    """
    basis = _compute_basis(x, v)
    prediction = basis @ params
    return prediction
