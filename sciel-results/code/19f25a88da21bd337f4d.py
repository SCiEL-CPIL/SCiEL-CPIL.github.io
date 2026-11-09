"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input x from [0.1, 6] to [-1, 1] to improve numerical stability
    # This helps BFGS converge more reliably for higher-order terms
    z = (x - 3.05) / 2.95
    
    # Use a 9th-degree polynomial basis (10 parameters)
    # This allows fitting a wide range of smooth functions (polynomials, 
    # exponentials, trigonometric functions) effectively.
    # 
    # Basis: [1, z, z^2, ..., z^9]
    #
    # We compute the basis matrix efficiently.
    # z is shape (N,)
    # powers is shape (10,)
    # basis is shape (N, 10)
    
    powers = np.arange(10)
    # np.power handles broadcasting: (N,) and (10,) -> (N, 10)
    basis = np.power(z, powers)
    
    # Linear combination
    y_pred = basis @ params
    
    return y_pred
# EVOLVE-BLOCK-END
