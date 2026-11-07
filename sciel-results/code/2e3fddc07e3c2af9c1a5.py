"""Symbolic regression: recover hidden univariate f on [-3,3] with 10 free params.

Strategy: Linear Radial Basis Functions (RBF).
Instead of a global trigonometric or polynomial basis, we use a local basis
of Gaussian functions centered at evenly spaced points across the domain [-3, 3].
The model is linear in the parameters (amplitudes), ensuring the MSE objective
is convex and BFGS converges reliably to the global least-squares minimum.

This structural change moves away from the Fourier series (current best) to a
local-interpolation-style basis, which may better capture non-periodic or
locally-varying features of the target function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 fixed centers evenly spaced in [-3, 3]
    # np.linspace(-3, 3, 10) gives: -3.0, -2.333, ..., 3.0
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width (standard deviation) for the Gaussian RBFs.
    # A width of 0.8 allows significant overlap between neighbors (spacing ~0.67),
    # ensuring smoothness and coverage across the domain.
    sigma = 0.8
    
    # Compute the RBF basis matrix
    # diff shape: (n_samples, 10)
    # centers shape: (1, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-diff^2 / (2 * sigma^2))
    # Pre-compute the constant factor
    kernel_const = -1.0 / (2.0 * sigma * sigma)
    
    basis = np.exp(kernel_const * diff * diff)
    
    # Linear combination: prediction = basis @ params
    # params is an array of length 10
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
