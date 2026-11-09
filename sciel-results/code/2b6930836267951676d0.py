import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Structural change: Fourier series basis with fixed orthogonal frequencies
    # up to the 5th harmonic. This leverages the natural orthogonality of
    # trigonometric functions on the domain [-2, 2] to ensure well-conditioned
    # gradients for BFGS without needing local RBF centers.
    
    # Parameters mapping:
    # p[0]: constant (DC term)
    # p[1]: cos(x) amplitude
    # p[2]: sin(x) amplitude
    # p[3]: cos(2x) amplitude
    # p[4]: sin(2x) amplitude
    # p[5]: cos(3x) amplitude
    # p[6]: sin(3x) amplitude
    # p[7]: cos(4x) amplitude
    # p[8]: sin(4x) amplitude
    # p[9]: cos(5x) amplitude
    
    # Compute Fourier terms
    # We use the standard Fourier basis with frequencies 1, 2, 3, 4, 5
    # For the domain [-2, 2], we use the natural frequencies
    
    y_hat = (p[0] * np.ones_like(x) + 
             p[1] * np.cos(x) + 
             p[2] * np.sin(x) +
             p[3] * np.cos(2 * x) + 
             p[4] * np.sin(2 * x) +
             p[5] * np.cos(3 * x) + 
             p[6] * np.sin(3 * x) +
             p[7] * np.cos(4 * x) + 
             p[8] * np.sin(4 * x) +
             p[9] * np.cos(5 * x))
             
    return y_hat
