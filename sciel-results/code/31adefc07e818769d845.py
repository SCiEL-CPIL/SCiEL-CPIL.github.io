"""Symbolic regression for a nonlinear oscillator using a parameterized RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Strategy:
# The previous approach used a fixed basis of sin, tanh, and polynomials, treating the
# problem as a linear regression in feature space. This limits the model to finding
# optimal weights for pre-defined shapes.
#
# This new strategy uses a **Parameterized Radial Basis Function (RBF)** network.
# The 10 parameters are split into:
# - 4 centers (c1, c2, c3, c4): Locations in the (x, v) space where the "bumps" are.
# - 2 widths (w1, w2): Control the spread of the Gaussian bumps.
# - 4 scales (s1, s4): Amplitude of each bump.
#
# Why this is different:
# 1. **Adaptive Geometry**: Instead of fixed sin/cos waves, the model places "atoms"
#    (Gaussians) exactly where the data requires complex structure.
# 2. **Non-linear Parameter Dependence**: The prediction is NOT linear in params.
#    This makes the landscape non-convex, but BFGS is a local optimizer that can
#    navigate these valleys if the initialization is reasonable (all ones).
# 3. **Interpolation Capability**: RBFs are universal approximators. With enough
#    centers, they can fit any smooth function. With only 4 centers, they act as a
#    flexible "blob" model that can capture asymmetric or localized nonlinearities
#    better than global sin/cos terms.
#
# Equation:
# y = s1 * exp(-w1 * ((x - c1)^2 + (v - c2)^2)) 
#   + s2 * exp(-w1 * ((x - c3)^2 + (v - c4)^2))
#   + s3 * exp(-w2 * ((x - c1)^2 + (v - c2)^2)) 
#   + s4 * exp(-w2 * ((x - c3)^2 + (v - c4)^2))
#
# Note: We reuse centers for different widths to save parameter budget, allowing
# for multi-scale features (sharp and smooth components) at the same locations.
# We also add a simple linear damping term if we have spare params? No, we must use
# exactly 10 params.
# Let's refine the parameter allocation for better coverage:
# p0, p1: Center 1 (x, v)
# p2, p3: Center 2 (x, v)
# p4, p5: Center 3 (x, v)
# p6: Width 1 (shared scale for sharpness)
# p7: Width 2 (shared scale for smoothness)
# p8: Scale 1 (amplitude for Center 1)
# p9: Scale 2 (amplitude for Center 2)
# Wait, 3 centers + 2 widths + 2 scales = 7 params. We have 3 left.
# Let's use:
# p0, p1: Center 1 (x, v)
# p2, p3: Center 2 (x, v)
# p4, p5: Center 3 (x, v)
# p6: Width (global width, ensures stability)
# p7: Scale 1
# p8: Scale 2
# p9: Scale 3
#
# Actually, a single global width is safer for BFGS starting from 1s.
# If width is 1, exp(-1 * dist^2) is a reasonable local feature.
# If scales are 1, the sum is a small value.
# The derivative with respect to centers is 0 at the start if x,v are not near 1,1?
# No, x is in [-0.5, 0.5], v in [-0.2, 0.3]. Centers start at 1.
# Distance is ~1. exp(-1) ~ 0.36. Gradients are non-zero. BFGS should work.

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict acceleration using a parameterized RBF model.
    
    params: [c1x, c1v, c2x, c2v, c3x, c3v, width, s1, s2, s3]
    """
    # Ensure 1D
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    
    # Unpack parameters
    c1x, c1v, c2x, c2v, c3x, c3v, width, s1, s2, s3 = params
    
    # Calculate squared distances from each center
    # (x - cx)^2 + (v - cv)^2
    dist1_sq = (x - c1x)**2 + (v - c1v)**2
    dist2_sq = (x - c2x)**2 + (v - c2v)**2
    dist3_sq = (x - c3x)**2 + (v - c3v)**2
    
    # Apply Gaussian kernel with shared width
    # Clip the exponent to prevent underflow issues, though exp handles small numbers well
    # width starts at 1. If width becomes very large, exp(-large) -> 0.
    # If width becomes very small, exp(-small) -> 1 (flat).
    # To keep it bounded and smooth, we use the raw width.
    
    # Use np.exp directly. For large negative exponents, it returns 0.
    k1 = np.exp(-width * dist1_sq)
    k2 = np.exp(-width * dist2_sq)
    k3 = np.exp(-width * dist3_sq)
    
    # Linear combination of RBFs
    pred = s1 * k1 + s2 * k2 + s3 * k3
    
    return pred
# EVOLVE-BLOCK-END
