import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Structural change: Combine a learnable sigmoid (for step/saturating behaviors)
    # with a low-order polynomial basis. This provides a different functional geometry
    # compared to pure Fourier or Chebyshev bases, potentially capturing non-smooth
    # features better.
    
    # Parameters mapping:
    # p[0]: polynomial constant
    # p[1]: polynomial x
    # p[2]: polynomial x^2
    # p[3]: polynomial x^3
    # p[4]: polynomial x^4
    # p[5]: sigmoid amplitude
    # p[6]: sigmoid center
    # p[7]: sigmoid steepness
    # p[8]: extra polynomial term x^5 (to balance degrees of freedom)
    # p[9]: extra polynomial term x^6
    
    # Compute polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x5 * x
    
    # Compute sigmoid term: amplitude * sigmoid(steepness * (x - center))
    # sigmoid(z) = 1 / (1 + exp(-z))
    # We use a stable version of sigmoid to avoid overflow
    z = p[7] * (x - p[6])
    
    # Stable sigmoid calculation
    # For large positive z, sigmoid(z) ~ 1
    # For large negative z, sigmoid(z) ~ 0
    # We can use np.where to handle cases, or just rely on numpy's stability if we clip
    # However, standard 1/(1+exp(-z)) can overflow if -z is large positive.
    # Let's use a safe sigmoid:
    # sigmoid(z) = 0.5 * (1 + tanh(z/2))
    # tanh is stable in numpy.
    sigmoid_term = 0.5 * (1.0 + np.tanh(0.5 * z))
    
    y_hat = (p[0] * np.ones_like(x) + 
             p[1] * x + 
             p[2] * x2 + 
             p[3] * x3 + 
             p[4] * x4 +
             p[5] * sigmoid_term +
             p[8] * x5 +
             p[9] * x6)
             
    return y_hat
