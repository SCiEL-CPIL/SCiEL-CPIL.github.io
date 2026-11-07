"""
Symbolic regression with 1 input.
Uses a degree 9 polynomial basis with scaled input for numerical stability.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a degree 9 polynomial with scaled input.
    
    The model is:
        f(x) = sum_{i=0}^{9} params[i] * (x/3.0)^i
    
    Scaling x by 1/3 maps the domain [-3, 3] to [-1, 1], which improves
    the conditioning of the optimization problem for BFGS, preventing
    large gradients and numerical instability in higher-order terms.
    """
    # Scale input to [-1, 1] for numerical stability
    z = x / 3.0
    
    # Compute powers of z: [1, z, z^2, ..., z^9]
    # Using vander for efficiency and clarity
    # np.vander(z, N=10, increasing=True) returns columns [z^0, z^1, ..., z^9]
    powers = np.vander(z, N=10, increasing=True)
    
    # Linear combination of the basis functions
    # powers shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    y = powers @ params
    
    return y
# EVOLVE-BLOCK-END
