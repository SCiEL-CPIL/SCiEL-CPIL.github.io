import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a 9th-degree polynomial with scaled input.
    Basis: 1, (x/scale), (x/scale)^2, ..., (x/scale)^9
    
    This is a structural change from the previous Gaussian RBF approach.
    Polynomials are global functions, which may better capture the overall trend
    of the target function with only 10 parameters, assuming the target is smooth.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale x to [~0, 1] to improve numerical conditioning for the basis functions.
    # Original range: [0.1, 6]. Scaling by 6.0 maps to [~0.016, 1.0].
    scale = 6.0
    x_scaled = x / scale
    
    # Create the monomial basis matrix of shape (N, 10)
    # columns are 1, x_s, x_s^2, ..., x_s^9
    # Using np.power for vectorized computation
    # We construct powers 0 to 9
    powers = np.arange(10)
    
    # x_scaled shape: (N,)
    # We want (N, 10). 
    # x_scaled[:, None] is (N, 1)
    # powers[None, :] is (1, 10)
    # Resulting shape: (N, 10)
    basis = np.power(x_scaled[:, None], powers[None, :])
    
    # Linear combination
    # If basis @ p produces non-finite values (due to numerical issues), 
    # we should ideally handle it, but the contract says non-finite error costs 1e30.
    # With scaling, values should be well-behaved (max 1^9 = 1).
    return basis @ p
