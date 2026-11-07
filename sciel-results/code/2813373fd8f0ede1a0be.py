"""Symbolic regression with 1 input(s) - Radial Basis Function (RBF) network with polynomial trend."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Radial Basis Function (RBF) network with 5 Gaussian kernels plus a linear trend.
    
    The 10 parameters are split as:
      p[0]: constant term (global offset)
      p[1]: linear coefficient (global slope)
      p[2..6]: 5 RBF centers (c_0 .. c_4)
      p[7..9]: 3 RBF widths (sigma_0 .. sigma_2)
    
    Wait, that's only 3 widths for 5 centers. Let me reconsider.
    
    With 10 params total, a clean split is:
      p[0]: constant
      p[1]: linear slope
      p[2..6]: 5 centers
      p[7..9]: 3 widths
    
    But we need 5 widths for 5 kernels. Alternative: use 4 kernels with 4 centers and 4 widths,
    plus constant, linear, and one extra param for a quadratic term.
    
    Let's use:
      p[0]: constant
      p[1]: linear coefficient
      p[2]: quadratic coefficient
      p[3..6]: 4 RBF centers
      p[7..9]: 3 RBF widths (use same width for kernel 4 as kernel 3, or use a default)
    
    Actually, a cleaner structural change: use 5 RBF kernels where the centers are learnable
    and the widths are fixed at a reasonable value, OR use 4 centers + 4 widths + 2 polynomial terms.
    
    Best structural choice for RBF: 
      p[0..4]: 5 centers
      p[5..9]: 5 widths
    
    This is a pure RBF with no explicit polynomial. But RBFs with Gaussian kernels can
    approximate constants and linear terms if the centers and widths are chosen appropriately.
    
    However, to ensure we can fit a linear trend easily, let's use:
      p[0]: constant
      p[1]: linear
      p[2..5]: 4 centers
      p[6..9]: 4 widths
    
    This gives 4 RBF kernels + polynomial trend. This is a structural change from the
    fixed-basis approach to a local-feature-interpolation approach.
    
    The model is:
      y(x) = p[0] + p[1]*x + sum_{i=0}^{3} exp(-(x - c_i)^2 / (2 * sigma_i^2))
    
    Wait, the amplitudes are fixed at 1. That's not flexible enough. We need learnable amplitudes too.
    
    With 10 params, we can't have 4 centers, 4 widths, AND 4 amplitudes. 
    So we fix the amplitudes to 1 and let the optimizer adjust centers and widths.
    This is a valid RBF structure where the "height" of each bump is controlled by
    the width (narrower = taller for a fixed integral, but here we fix amplitude=1).
    
    Alternative: use 5 centers with fixed widths, and learn the amplitudes.
      p[0..4]: 5 amplitudes
      p[5..9]: 5 centers
    Widths fixed at 0.5.
    
    This is actually a good structure: learnable locations and heights of 5 Gaussian bumps.
    The polynomial trend is captured by the combination of bumps.
    
    Let's go with:
      p[0..4]: amplitudes a_i
      p[5..9]: centers c_i
      widths fixed at 0.5
    
    y(x) = sum_{i=0}^{4} a_i * exp(-(x - c_i)^2 / (2 * 0.5^2))
    
    This is a clean RBF structure. The optimizer can place bumps anywhere in [-2, 2]
    and adjust their heights.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = params
    
    # Clip x to avoid numerical issues
    x_c = np.clip(x, -50, 50)
    
    # Extract parameters
    amplitudes = p[0:5]  # a_0 .. a_4
    centers = p[5:10]    # c_0 .. c_4
    
    # Fixed widths
    sigma = 0.5
    sigma2 = 2.0 * sigma * sigma  # 2 * sigma^2
    
    # Compute RBF sum
    y = np.zeros_like(x_c)
    for i in range(5):
        diff = x_c - centers[i]
        # Clip the exponent argument to avoid overflow/underflow issues
        exponent = -(diff * diff) / sigma2
        exponent = np.maximum(exponent, -100)
        y += amplitudes[i] * np.exp(exponent)
    
    return y
# EVOLVE-BLOCK-END
