"""
Symbolic regression using an orthogonal Chebyshev polynomial basis.
The hidden function is approximated by a linear combination of Chebyshev polynomials
of the first kind, T_k(x), mapped to the domain [-2, 2].
y(x) = sum_{k=0}^{9} c_k * T_k(x/2)

This approach is materially different from the rational function approach because:
1. The model is linear in the parameters (c_k), making the loss surface convex.
2. BFGS on a convex problem with a well-conditioned design matrix (Chebyshev)
   converges to the global minimum reliably, avoiding local minima and instability.
3. Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight 1/sqrt(1-x^2),
   which provides a stable and efficient basis for approximating smooth functions on a bounded interval.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Start timer for safety margin
_START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s buffer before 150s kill

def _chebyshev_basis(x: np.ndarray, degree: int = 9) -> np.ndarray:
    """
    Compute Chebyshev polynomials of the first kind T_k(t) for t in [-1, 1].
    Input x is in [-2, 2]. We map x -> t = x/2.
    Returns a matrix of shape (N, degree+1) where column k is T_k(t).
    """
    t = x / 2.0  # Map to [-1, 1]
    
    # Preallocate
    N = x.size
    basis = np.zeros((N, degree + 1), dtype=np.float64)
    
    if N == 0:
        return basis
        
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    if degree >= 1:
        # T_1(t) = t
        basis[:, 1] = t
        
    for k in range(2, degree + 1):
        # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients (c_0, ..., c_9)
    Returns: predicted values
    """
    if x.size == 0:
        return np.array([])
    
    # Check time limit
    if time.time() - _START_TIME > TIME_LIMIT:
        # Return a simple fallback if time is up (should not happen in normal flow)
        return np.zeros_like(x)

    # Ensure params is a flat array of length 10
    params = np.asarray(params, dtype=np.float64).flatten()
    if params.size < 10:
        # Pad with zeros if necessary, though evaluator provides 10
        params = np.pad(params, (0, 10 - params.size))
    else:
        params = params[:10]

    # Compute the Chebyshev basis matrix
    # Degree 9 uses all 10 parameters
    basis = _chebyshev_basis(x, degree=9)
    
    # The prediction is the dot product of the basis and the parameters
    # basis is (N, 10), params is (10,)
    y_pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(y_pred)):
        # Fallback to zero if numerical instability occurs
        y_pred = np.zeros_like(x)
        
    return y_pred
# EVOLVE-BLOCK-END
