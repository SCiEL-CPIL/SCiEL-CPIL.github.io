"""
Fourier series basis with 10 parameters.
This maps the input domain [0.1, 6] to [-1, 1] and uses a truncated
Fourier series as the basis. The basis functions are orthogonal on the
interval, which results in a well-conditioned design matrix. This
stability allows BFGS to reliably converge to the optimal linear
coefficients, avoiding the numerical issues associated with high-degree
polynomials.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series model with 10 parameters.
    params[0]: constant term
    params[1]: cos(pi*t)
    params[2]: sin(pi*t)
    params[3]: cos(2*pi*t)
    params[4]: sin(2*pi*t)
    params[5]: cos(3*pi*t)
    params[6]: sin(3*pi*t)
    params[7]: cos(4*pi*t)
    params[8]: sin(4*pi*t)
    params[9]: cos(5*pi*t)
    
    1. Map x from [0.1, 6] to t in [-1, 1].
       t = (2 * (x - 0.1) - 5.9) / 5.9
       
    2. Compute Fourier basis functions.
       We use 5 cosine terms and 4 sine terms plus a constant.
       This provides a mix of even and odd symmetry components.
       
    3. Prediction is the dot product of params and the basis matrix.
    """
    # Map x to t in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    range_x = x_max - x_min  # 5.9
    
    # t = (2 * (x - x_min) - range_x) / range_x
    t = (2.0 * (x - x_min) - range_x) / range_x
    
    # Number of parameters
    n_params = 10
    
    # Initialize basis matrix
    basis = np.empty((x.shape[0], n_params), dtype=np.float64)
    
    # Constant term
    basis[:, 0] = 1.0
    
    # Harmonics k = 1 to 4: cos(k*pi*t) and sin(k*pi*t)
    # This uses 8 parameters (indices 1 to 8)
    for k in range(1, 5):
        angle = k * np.pi * t
        basis[:, 2*k - 1] = np.cos(angle)
        basis[:, 2*k] = np.sin(angle)
        
    # 5th Harmonic: only cos(5*pi*t) to fill the 10th parameter (index 9)
    # This is slightly asymmetric but ensures we use all 10 params.
    # Alternatively, we could drop one sine term, but adding the next cosine
    # is a standard way to extend the series if parameter count is odd after DC.
    basis[:, 9] = np.cos(5 * np.pi * t)
    
    # Prediction
    prediction = basis @ params
    
    return prediction
