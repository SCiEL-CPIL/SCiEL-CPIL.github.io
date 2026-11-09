"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a Fourier series basis on the mapped interval [-1, 1].
    
    The basis consists of 1, sin(pi*t), cos(pi*t), sin(2*pi*t), cos(2*pi*t), ...
    This provides a set of orthogonal, bounded functions that are well-conditioned
    for BFGS optimization, avoiding the potential instability of high-degree
    polynomial bases on this domain.
    """
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # We have 10 parameters.
    # Let's assign them to the first 10 terms of the Fourier series.
    # Term 0: 1 (DC component)
    # Term 1: sin(pi * t)
    # Term 2: cos(pi * t)
    # Term 3: sin(2 * pi * t)
    # Term 4: cos(2 * pi * t)
    # ...
    
    # Precompute basis functions
    # Shape (10, n_points)
    basis = np.zeros((10, n_points))
    
    # DC component
    basis[0, :] = 1.0
    
    # We need 9 more terms.
    # Let's fill them with sin and cos of increasing frequencies.
    # freq 1: sin, cos (indices 1, 2)
    # freq 2: sin, cos (indices 3, 4)
    # freq 3: sin, cos (indices 5, 6)
    # freq 4: sin, cos (indices 7, 8)
    # freq 5: sin (index 9)
    
    for i in range(1, 10):
        # Determine frequency and type
        # i=1 -> freq 1, sin
        # i=2 -> freq 1, cos
        # i=3 -> freq 2, sin
        # i=4 -> freq 2, cos
        # ...
        # freq = (i + 1) // 2
        # is_sin = (i % 2 == 1)
        
        freq = (i + 1) // 2
        if i % 2 == 1:
            basis[i, :] = np.sin(np.pi * freq * t)
        else:
            basis[i, :] = np.cos(np.pi * freq * t)
            
    # Prediction is the linear combination
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
