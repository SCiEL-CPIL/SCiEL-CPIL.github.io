"""Rewritten program for symbolic regression using Orthonormal Legendre basis for stability."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    The evaluator fits `params` (length 10) via BFGS starting from all ones,
    minimizing training MSE. We design a flexible, smooth model with 10 parameters.
    
    Key Improvement:
    Instead of raw powers $x, x^2, x^3$, we use an Orthonormal Legendre basis 
    mapped to the domain [-3, 3]. This improves the condition number of the 
    design matrix, stabilizing BFGS optimization.
    
    Parameter allocation (10 total):
      params[0]: Coefficient for P_0(t) (constant)
      params[1]: Coefficient for P_1(t) (linear in t)
      params[2]: Coefficient for P_2(t) (quadratic in t)
      params[3]: Coefficient for P_3(t) (cubic in t)
      params[4]: Amplitude of sin(freq1 * x)
      params[5]: Frequency of sin(freq1 * x)
      params[6]: Amplitude of cos(freq2 * x)
      params[7]: Frequency of cos(freq2 * x)
      params[8]: Rate of exp(rate * x)
      params[9]: Amplitude of exp(rate * x)
      
    Where t = x / 3.0 maps x in [-3, 3] to t in [-1, 1].
    P_n(t) are standard Legendre polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    amp_sin = params[4]
    freq_sin = params[5]
    amp_cos = params[6]
    freq_cos = params[7]
    rate_exp = params[8]
    amp_exp = params[9]
    
    # Map x from [-3, 3] to t in [-1, 1] for Legendre polynomials
    t = x / 3.0
    
    # Legendre Polynomials P_n(t)
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t^2 - 1)
    # P_3(t) = 0.5 * (5*t^3 - 3*t)
    
    p0 = np.ones_like(t)
    p1 = t
    p2 = 0.5 * (3.0 * t**2 - 1.0)
    p3 = 0.5 * (5.0 * t**3 - 3.0 * t)
    
    # Polynomial component using orthonormal basis
    # Note: Standard Legendre polynomials are orthogonal but not normalized to unit L2 norm on [-1,1].
    # However, they are much better conditioned than raw powers.
    # For BFGS stability, this is a significant improvement over x, x^2, x^3.
    y_legendre = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3
    
    # Trigonometric terms (using original x to allow any frequency)
    # Clip frequencies for numerical stability
    freq_sin = np.clip(freq_sin, -20, 20)
    freq_cos = np.clip(freq_cos, -20, 20)
    
    y_sin = amp_sin * np.sin(freq_sin * x)
    y_cos = amp_cos * np.cos(freq_cos * x)
    
    # Exponential term
    rate_exp = np.clip(rate_exp, -10, 10)
    exp_arg = rate_exp * x
    exp_arg = np.clip(exp_arg, -30, 30)
    y_exp = amp_exp * np.exp(exp_arg)
    
    # Combine all components
    result = y_legendre + y_sin + y_cos + y_exp
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
