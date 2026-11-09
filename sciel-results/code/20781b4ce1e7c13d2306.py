"""Symbolic regression with Fourier series basis over [0.1, 6]."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series expansion with 10 fixed orthogonal basis functions.
    
    The domain is [0.1, 6]. We use a shifted and scaled coordinate t in [0, 1]
    so that the Fourier basis has integer frequencies over [0, 1], ensuring
    orthogonality on the training domain.
    
    params[0..9] are coefficients for the basis functions.
    """
    # Map x from [0.1, 6] to t in [0, 1]
    # t = (x - 0.1) / (6 - 0.1) = (x - 0.1) / 5.9
    t = (x - 0.1) / 5.9
    
    # Build the Fourier basis: 1, cos(2*pi*t), sin(2*pi*t), cos(4*pi*t), sin(4*pi*t), ...
    # With 10 params, we can have 5 cosine terms and 5 sine terms, or mix.
    # Let's use: 1, cos(2πt), sin(2πt), cos(4πt), sin(4πt), cos(6πt), sin(6πt), cos(8πt), sin(8πt), cos(10πt)
    # That's 10 basis functions.
    
    # Precompute angles
    two_pi = 2.0 * np.pi
    
    # Basis functions
    # b0 = 1
    # b1 = cos(2*pi*t)
    # b2 = sin(2*pi*t)
    # b3 = cos(4*pi*t)
    # b4 = sin(4*pi*t)
    # b5 = cos(6*pi*t)
    # b6 = sin(6*pi*t)
    # b7 = cos(8*pi*t)
    # b8 = sin(8*pi*t)
    # b9 = cos(10*pi*t)
    
    result = np.zeros_like(x, dtype=np.float64)
    
    result += params[0]
    result += params[1] * np.cos(two_pi * t)
    result += params[2] * np.sin(two_pi * t)
    result += params[3] * np.cos(4.0 * np.pi * t)
    result += params[4] * np.sin(4.0 * np.pi * t)
    result += params[5] * np.cos(6.0 * np.pi * t)
    result += params[6] * np.sin(6.0 * np.pi * t)
    result += params[7] * np.cos(8.0 * np.pi * t)
    result += params[8] * np.sin(8.0 * np.pi * t)
    result += params[9] * np.cos(10.0 * np.pi * t)
    
    return result
