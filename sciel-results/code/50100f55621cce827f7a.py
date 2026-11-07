"""
Fourier series basis for symbolic regression on [0.1, 6].
Uses a mix of sine and cosine terms with frequencies scaled to the domain,
plus a constant and a linear term. The model is linear in params (weights),
so BFGS converges to the global optimum for this fixed basis.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series model with 10 parameters:
    - params[0]: constant (DC) term
    - params[1]: linear term coefficient (x)
    - params[2:10]: coefficients for 8 Fourier terms (4 cos + 4 sin)
    
    Frequencies are chosen to span the domain [0.1, 6] with increasing
    angular frequencies. The basis is linear in params, so BFGS finds
    the global optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # params[0]: constant
    # params[1]: linear term
    const = params[0]
    linear = params[1] * x
    
    # Fourier terms: 4 cosine + 4 sine
    # Frequencies scaled so that k*pi covers roughly the domain
    # Domain length is ~5.9, so fundamental frequency ~ pi/5.9
    # Use k = 1, 2, 3, 4 for both sin and cos
    k_values = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Base frequency: pi / domain_length, domain_length = 6 - 0.1 = 5.9
    base_freq = np.pi / 5.9
    
    # Compute all Fourier terms: shape (n, 8)
    # cos(k * base_freq * x) for k in k_values, then sin(k * base_freq * x)
    angles = np.outer(x, k_values * base_freq)  # (n, 4)
    
    cos_terms = np.cos(angles)  # (n, 4)
    sin_terms = np.sin(angles)  # (n, 4)
    
    # params[2:6] for cos terms, params[6:10] for sin terms
    cos_weights = params[2:6]
    sin_weights = params[6:10]
    
    prediction = const + linear + cos_terms @ cos_weights + sin_terms @ sin_weights
    
    return prediction
