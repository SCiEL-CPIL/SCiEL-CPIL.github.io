"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Adaptive Nonlinear Basis (Gaussian RBFs + Polynomials).
Instead of a fixed linear basis (like Chebyshev), we use a nonlinear model where
part of the parameters define the basis functions themselves (Gaussian centers and
widths). This allows the model to adapt its shape to the hidden function's structure.

Model:
f(x) = sum_{i=0}^{4} c_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2)) + sum_{j=0}^{4} p_j * (x/3)^j

Parameters (length 10):
0-4: c_i (coefficients for Gaussians)
5-7: mu_i (centers for 3 Gaussians, fixed for the other 2 to reduce complexity)
8-9: sigma_i (widths for 3 Gaussians, fixed for the other 2)
Wait, we have 10 params. Let's use:
0: c0, 1: c1, 2: c2, 3: c3, 4: c4 (5 coeffs)
5: mu0, 6: mu1, 7: mu2 (3 centers)
8: sigma0, 9: sigma1 (2 widths)
Actually, let's make it more symmetric and robust.
Let's use 5 Gaussians with fixed centers at Chebyshev nodes? No, that's linear.
Let's use 5 Gaussians with 5 centers and 5 widths? That's 15 params. We only have 10.

Revised Plan:
Use a hybrid:
- 3 Adaptive Gaussians: 3 centers, 3 widths, 3 coeffs = 9 params.
- 1 Global Linear Term: 1 param.
Total = 10 params.

Parameters:
0: c0 (coeff for G0)
1: c1 (coeff for G1)
2: c2 (coeff for G2)
3: mu0 (center for G0)
4: mu1 (center for G1)
5: mu2 (center for G2)
6: sigma0 (width for G0)
7: sigma1 (width for G1)
8: sigma2 (width for G2)
9: p0 (linear slope)

This allows local adaptation via the Gaussians and a global trend via the linear term.
BFGS is used to optimize these 10 parameters. The loss is non-convex, but BFGS
starting from ones might find a good local minimum if the landscape is smooth.
To improve robustness, we can add a small regularization or ensure numerical stability.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Nonlinear Basis: 3 Gaussians + 1 Linear Term.
    
    Params:
    p[0]: c0
    p[1]: c1
    p[2]: c2
    p[3]: mu0
    p[4]: mu1
    p[5]: mu2
    p[6]: sigma0
    p[7]: sigma1
    p[8]: sigma2
    p[9]: p_linear
    
    Model:
    y = c0*exp(-(x-mu0)^2/(2*sigma0^2)) + c1*exp(-(x-mu1)^2/(2*sigma1^2)) + 
        c2*exp(-(x-mu2)^2/(2*sigma2^2)) + p_linear*x
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Extract parameters
    c0, c1, c2 = p[0], p[1], p[2]
    mu0, mu1, mu2 = p[3], p[4], p[5]
    s0, s1, s2 = p[6], p[7], p[8]
    p_lin = p[9]
    
    # Ensure sigmas are positive and not too small to avoid division by zero/overflow
    # Use softplus or abs + epsilon. BFGS might try negative sigmas.
    # sigma = abs(s) + 1e-3
    # But abs is not differentiable at 0. Use s^2 + 1e-3? Or just clip?
    # Let's use s^2 + 1e-3 to keep it smooth and positive.
    sig0 = s0*s0 + 1e-3
    sig1 = s1*s1 + 1e-3
    sig2 = s2*s2 + 1e-3
    
    # Calculate Gaussian terms
    # exp(-(x-mu)^2 / (2*sigma^2))
    # Note: sigma here is the standard deviation in the exponent denominator 2*sigma^2
    # If we want sigma to be the width parameter, let's define the exponent as -(x-mu)^2 / (2 * sigma_param^2)
    
    term0 = c0 * np.exp(-(x_flat - mu0)**2 / (2.0 * sig0))
    term1 = c1 * np.exp(-(x_flat - mu1)**2 / (2.0 * sig1))
    term2 = c2 * np.exp(-(x_flat - mu2)**2 / (2.0 * sig2))
    
    term_lin = p_lin * x_flat
    
    result = term0 + term1 + term2 + term_lin
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to a safe value if numerics fail
        result = np.zeros_like(x_flat)
        
    return result.reshape(orig_shape)
