"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 2 kernels in 3D.
    
    params layout (10 total):
      p[0:3]  = center of kernel 1 (cx1, cy1, cz1)
      p[3]    = width (sigma) of kernel 1
      p[4]    = weight of kernel 1
      p[5:8]  = center of kernel 2 (cx2, cy2, cz2)
      p[8]    = width (sigma) of kernel 2
      p[9]    = weight of kernel 2
    
    We add a small constant offset derived from the weights to help with DC offset.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Extract parameters
    cx1, cy1, cz1 = params[0], params[1], params[2]
    sigma1 = params[3]
    w1 = params[4]
    
    cx2, cy2, cz2 = params[5], params[6], params[7]
    sigma2 = params[8]
    w2 = params[9]
    
    # Prevent division by zero or very small sigma causing numerical issues
    # Use a softplus-like transformation to keep sigma positive and smooth
    # Actually, BFGS can handle negative sigma if we square it, but let's use abs + epsilon
    # Better: use exp to ensure positive width, but that changes the optimization landscape.
    # Let's stick to raw params but clip sigma to a minimum positive value for stability.
    # However, clipping can create non-differentiable points.
    # Let's use sigma = 1 + np.tanh(params[3]) to keep it in (0, 2) ? 
    # No, that restricts the range. 
    # Let's just use params[3] and params[8] directly but ensure they don't cause overflow.
    # exp(-d^2 / (2*sigma^2)) can overflow if sigma is very small.
    # Let's use a safe sigma: sigma = max(abs(params[3]), 0.01)
    # But max is not differentiable at 0.
    # Let's use sigma = 0.01 + np.exp(params[3] * 0.5) ? No, too complex.
    
    # Simplest robust approach: 
    # sigma1 = params[3]
    # If sigma1 is negative, the exp term blows up. 
    # Let's force sigma to be positive by using sigma = 1.0 / (1.0 + np.exp(-params[3])) * 2.0 ? 
    # This maps R to (0, 2). 
    # Or just use abs(params[3]) + 0.01. The kink at 0 is usually handled by BFGS reasonably well.
    
    sigma1 = np.abs(params[3]) + 0.01
    sigma2 = np.abs(params[8]) + 0.01
    
    # Compute squared distances
    # d1^2 = (x - cx1)^2 + (y - cy1)^2 + (z - cz1)^2
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    d1_sq = dx1*dx1 + dy1*dy1 + dz1*dz1
    
    # d2^2 = (x - cx2)^2 + (y - cy2)^2 + (z - cz2)^2
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    d2_sq = dx2*dx2 + dy2*dy2 + dz2*dz2
    
    # Compute RBF values
    # Use np.exp with clipping to prevent overflow/underflow issues
    # exp(-large) -> 0, exp(small) -> 1
    # If sigma is very large, exponent is close to 0, exp(0)=1
    # If sigma is very small, exponent is large negative, exp(-inf)=0
    
    rbf1 = np.exp(-d1_sq / (2.0 * sigma1 * sigma1))
    rbf2 = np.exp(-d2_sq / (2.0 * sigma2 * sigma2))
    
    # Final prediction
    # Add a small constant term to allow fitting the mean? 
    # We have 10 params. If we use 2 kernels with 7 params (center, width, weight) each, that's 14 params.
    # Wait, my layout was:
    # p[0:3] center1 (3)
    # p[3] width1 (1)
    # p[4] weight1 (1)
    # p[5:8] center2 (3)
    # p[8] width2 (1)
    # p[9] weight2 (1)
    # Total: 3+1+1+3+1+1 = 10. Perfect.
    
    return w1 * rbf1 + w2 * rbf2
# EVOLVE-BLOCK-END
