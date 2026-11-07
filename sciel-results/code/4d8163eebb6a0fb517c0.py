"""Nonlinear oscillator acceleration via Gaussian Process-inspired kernel basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a sum of smooth
    Gaussian bumps in the (t, x, v) state space.
    
    The 10 parameters define:
    - 3 Gaussian centers (in t, x, v dimensions)
    - 3 Gaussian scales (inverse bandwidths)
    - 3 Gaussian weights
    - 1 global offset
    
    This provides a flexible, non-parametric-like approximation to the
    acceleration field without assuming specific polynomial/rational structure.
    """
    # Extract parameters
    ct, cx, cv = params[0], params[1], params[2]  # centers
    st, sx, sv = params[3], params[4], params[5]  # scales (inverse bandwidths)
    wt, wx, wv = params[6], params[7], params[8]  # weights
    offset = params[9]  # global offset
    
    # Ensure scales are positive to avoid numerical issues
    # Use softplus-like: abs + small epsilon
    st = np.abs(st) + 1e-6
    sx = np.abs(sx) + 1e-6
    sv = np.abs(sv) + 1e-6
    
    # Compute Gaussian bumps
    # Each bump is centered at (ct, cx, cv) with independent scales
    dt = (t - ct) / st
    dx = (x - cx) / sx
    dv = (v - cv) / sv
    
    # Sum of three weighted Gaussians (one emphasizing each dimension)
    bump_t = wt * np.exp(-0.5 * dt**2)
    bump_x = wx * np.exp(-0.5 * dx**2)
    bump_v = wv * np.exp(-0.5 * dv**2)
    
    # Combine: the acceleration is a weighted sum of these bumps plus offset
    # We use a product structure to allow coupling: the total is the sum
    # of three terms, each capturing dependence on one primary variable
    return bump_t + bump_x + bump_v + offset
# EVOLVE-BLOCK-END
