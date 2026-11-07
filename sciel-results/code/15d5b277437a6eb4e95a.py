"""Symbolic regression using Chebyshev polynomial basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis with 10 parameters.
    
    This approach maps the input domain [0.1, 6] to the standard Chebyshev domain [-1, 1].
    It then constructs a basis of the first 10 Chebyshev polynomials T_0(t) through T_9(t).
    The parameters are the coefficients for these polynomials.
    
    Chebyshev polynomials provide a well-conditioned basis for approximating smooth
    functions on a bounded interval, often outperforming standard power bases
    (1, x, x^2, ...) due to their orthogonality properties and minimax error characteristics.
    
    params[0:10] are coefficients for T_0(t) through T_9(t).
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Define domain bounds
    x_min = 0.1
    x_max = 6.0
    
    # Map x from [x_min, x_max] to t in [-1, 1]
    # t = 2 * (x - x_min) / (x_max - x_min) - 1
    t = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    # Clip t to [-1, 1] to handle any numerical overshoots, though mapping is linear
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    basis_list = [T0, T1]
    
    for n in range(2, 10):
        Tn = 2.0 * t * T1 - basis_list[-2]
        basis_list.append(Tn)
        T1 = Tn
        
    # Stack into matrix: shape (n, 10)
    basis = np.column_stack(basis_list)
    
    # params is shape (10,)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Predict: (n, 10) @ (10,) -> (n,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
