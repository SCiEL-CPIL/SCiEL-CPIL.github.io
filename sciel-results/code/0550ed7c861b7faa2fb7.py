"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-pi, pi] for standard Fourier basis
    # x is in [-2, 2], so t = pi * x / 2 is in [-pi, pi]
    t = np.pi * x / 2.0
    
    # Construct Fourier basis matrix
    # Frequencies: 1, 2, 3, 4, 5
    # Basis vectors: sin(t), cos(t), sin(2t), cos(2t), ..., sin(5t), cos(5t)
    # Total 10 parameters
    
    # params shape: (10,)
    # We map:
    # p0 -> sin(t)
    # p1 -> cos(t)
    # p2 -> sin(2t)
    # p3 -> cos(2t)
    # p4 -> sin(3t)
    # p5 -> cos(3t)
    # p6 -> sin(4t)
    # p7 -> cos(4t)
    # p8 -> sin(5t)
    # p9 -> cos(5t)
    
    # Vectorized construction
    # We can create a matrix of shape (N, 10)
    
    # Calculate sines and cosines for all frequencies
    # freqs = [1, 2, 3, 4, 5]
    # t is shape (N,)
    # We want to stack sin(k*t) and cos(k*t)
    
    # Let's build the basis column by column or using broadcasting
    # t[:, np.newaxis] is (N, 1)
    # freqs = np.arange(1, 6) # (5,)
    # angles = t[:, np.newaxis] * freqs[np.newaxis, :] # (N, 5)
    # sines = np.sin(angles) # (N, 5)
    # cosines = np.cos(angles) # (N, 5)
    # Interleave or stack?
    # If we interleave: [sin(1), cos(1), sin(2), cos(2), ...]
    # This matches the parameter mapping above.
    
    angles = t[:, np.newaxis] * np.arange(1, 6)
    sines = np.sin(angles)
    cosines = np.cos(angles)
    
    # Interleave sines and cosines
    # sines: (N, 5), cosines: (N, 5)
    # Reshape to (N, 2, 5) and transpose to (N, 5, 2) then flatten?
    # Or just stack and reshape.
    # Let's use a trick:
    # basis = np.stack([sines, cosines], axis=2).reshape(x.shape[0], 10)
    # This results in [sin1, cos1, sin2, cos2, ...]
    
    basis = np.stack([sines, cosines], axis=2).reshape(x.shape[0], 10)
    
    # Dot product with params
    return basis @ params
# EVOLVE-BLOCK-END
