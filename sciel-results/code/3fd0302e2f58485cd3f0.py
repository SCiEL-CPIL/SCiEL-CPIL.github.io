"""
Strategy: RBF (Radial Basis Function) network with Gaussian kernels.

This is a structural change from global polynomial to local kernel regression.
The 10 parameters are divided into:
- 3 centers (cx1, cy1, cx2, cy2, cx3, cy3) -> actually 6 params for 3 centers in 2D
- 1 shared width (sigma)
- 3 weights (w1, w2, w3)
Total: 6 + 1 + 3 = 10 params.

Wait, that's 6+1+3 = 10. Perfect.

The model is:
a = sum_{i=1}^{3} w_i * exp(-((x - cx_i)^2 + (v - cy_i)^2) / (2 * sigma^2))

This allows local adaptation to the nonlinear oscillator dynamics.
Gaussian RBFs are smooth and can capture local structure that global polynomials miss.

To help BFGS:
- Scale inputs to roughly [-1, 1]
- Initialize centers within the training range
- The width sigma controls the local vs global nature
"""
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts acceleration using an RBF network with 3 Gaussian kernels.
    
    Inputs:
        x: position
        v: velocity
        params: array of 10 parameters
            [cx1, cy1, cx2, cy2, cx3, cy3, sigma, w1, w2, w3]
    
    Model:
        a = w1 * exp(-((x-cx1)^2 + (v-cy1)^2) / (2*sigma^2))
          + w2 * exp(-((x-cx2)^2 + (v-cy2)^2) / (2*sigma^2))
          + w3 * exp(-((x-cx3)^2 + (v-cy3)^2) / (2*sigma^2))
    """
    p = params
    
    # Extract parameters
    cx1, cy1, cx2, cy2, cx3, cy3, sigma, w1, w2, w3 = p
    
    # Ensure sigma is positive to avoid division by zero or negative variance
    sigma = np.abs(sigma) + 1e-8
    
    # Scale inputs to improve condition number
    # Training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    x_scale = 0.6
    v_scale = 0.25
    
    x_s = x / x_scale
    v_s = v / v_scale
    
    # Scale centers and sigma to match scaled space
    cx1_s = cx1 / x_scale
    cy1_s = cy1 / v_scale
    cx2_s = cx2 / x_scale
    cy2_s = cy2 / v_scale
    cx3_s = cx3 / x_scale
    cy3_s = cy3 / v_scale
    sigma_s = sigma / 0.42  # approximate RMS of the scaling factors
    
    # Compute squared distances
    d1_sq = (x_s - cx1_s)**2 + (v_s - cy1_s)**2
    d2_sq = (x_s - cx2_s)**2 + (v_s - cy2_s)**2
    d3_sq = (x_s - cx3_s)**2 + (v_s - cy3_s)**2
    
    # Compute Gaussian kernels
    # Use 2*sigma^2 in denominator
    denom = 2.0 * sigma_s**2
    
    k1 = np.exp(-d1_sq / denom)
    k2 = np.exp(-d2_sq / denom)
    k3 = np.exp(-d3_sq / denom)
    
    # Combine with weights
    result = w1 * k1 + w2 * k2 + w3 * k3
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
