"""
Strategy: Locally Adaptive Gaussian Basis (Neural Network-like)
Instead of a fixed global basis, we use 5 Gaussian bumps with learnable centers and widths.
This allows the model to fit local features more accurately than a fixed polynomial/trig basis.
The parameters are:
- p[0:5]: Amplitudes of the 5 Gaussians
- p[5:9]: Centers of the first 4 Gaussians (the 5th is fixed at 0 to ensure central coverage)
- p[9]: Width scale (shared for all Gaussians to maintain stability and reduce parameters)
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 5 adaptive Gaussians.
    y = sum_{i=1}^{5} A_i * exp(-0.5 * ((x - c_i) / sigma)^2)
    """
    # Ensure x is a 1D float array
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Pad if necessary (though evaluator provides 10)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
        
    # Extract parameters
    # Amplitudes
    A1, A2, A3, A4, A5 = p[0], p[1], p[2], p[3], p[4]
    
    # Centers (learnable for first 4, fixed at 0 for the 5th)
    c1, c2, c3, c4 = p[5], p[6], p[7], p[8]
    c5 = 0.0
    
    # Width (shared, learnable). 
    # We add a small epsilon to prevent division by zero if sigma -> 0.
    # The sign doesn't matter for the square, but we use abs to be safe.
    sigma = np.abs(p[9]) + 1e-6
    
    # Compute Gaussian terms
    # Using vectorized operations for speed
    # Term 1
    t1 = A1 * np.exp(-0.5 * ((x - c1) / sigma) ** 2)
    # Term 2
    t2 = A2 * np.exp(-0.5 * ((x - c2) / sigma) ** 2)
    # Term 3
    t3 = A3 * np.exp(-0.5 * ((x - c3) / sigma) ** 2)
    # Term 4
    t4 = A4 * np.exp(-0.5 * ((x - c4) / sigma) ** 2)
    # Term 5
    t5 = A5 * np.exp(-0.5 * ((x - c5) / sigma) ** 2)
    
    # Sum
    y = t1 + t2 + t3 + t4 + t5
    
    return y

# EVOLVE-BLOCK-END
