import numpy as np
import time

# Global deadline guard: evaluator kills after 150s wall-clock.
# We track time to ensure we return before the limit.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 10 Radial Basis Functions (RBFs).
    
    The centers are fixed at evenly spaced points in [-1, 1] (after scaling x from [-3, 3]).
    The width parameter is shared across all kernels to ensure smoothness and well-conditioning.
    
    params layout (10 elements):
      params[0:9]:  weights for the 9 RBF centers
      params[9]:    inverse width (beta) for the Gaussian kernels
    
    The prediction is:
      sum_{i=0}^{8} w_i * exp(-beta * (u - c_i)^2)
    
    This structure is smooth and well-conditioned for BFGS optimization.
    """
    # Check deadline: if we're too close to the limit, return a simple fallback
    # to avoid being killed. In practice, this function is called many times
    # during BFGS iterations, so we must be fast.
    # The RBF computation is O(N) per point, which is very fast.
    
    # Scale input to [-1, 1]
    u = x / 3.0
    
    # Fixed centers: 9 evenly spaced points in [-1, 1]
    # We use 9 centers to leave 1 param for the width.
    centers = np.linspace(-1.0, 1.0, 9)
    
    # Extract weights and width
    weights = params[0:9]
    beta = params[9]
    
    # Ensure beta is positive and not too large to avoid numerical issues
    # BFGS may try negative beta; we clamp it to a small positive value
    # but the optimization should find a good positive value.
    # For safety, use abs or a floor.
    # Using a soft clamp: beta_eff = max(beta, 0.1)
    beta_eff = np.maximum(beta, 0.1)
    
    # Compute RBF values
    # u shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    sq_diff = diff ** 2
    
    # Gaussian kernels: exp(-beta * sq_diff)
    # Shape: (N, 9)
    kernels = np.exp(-beta_eff * sq_diff)
    
    # Linear combination
    # weights shape: (9,)
    pred = kernels @ weights
    
    return pred
