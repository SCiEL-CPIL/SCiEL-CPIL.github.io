import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

START_TIME = time.time()
DEADLINE = START_TIME + 140.0

# Strategy: Radial Basis Function (RBF) Network with Fixed Centers and Widths.
# We use 10 Gaussian RBFs. The parameters are the weights of these 10 RBFs.
# f(x) = sum_{i=0}^{9} w_i * exp(-alpha * (x - c_i)^2)
# Since alpha and c_i are fixed, this is a linear model in w.
# This is structurally different from B-Splines. RBFs are globally smooth but locally responsive.
# B-Splines are piecewise polynomial. RBFs might better capture functions with localized variations
# or "bumps" that B-Splines might smooth out too much or require more knots to capture.

# Define centers uniformly across the domain [0.1, 6]
DOMAIN_MIN = 0.1
DOMAIN_MAX = 6.0
N_CENTERS = 10
centers = np.linspace(DOMAIN_MIN, DOMAIN_MAX, N_CENTERS)

# Choose a width parameter (sigma) for the Gaussian.
# If sigma is too small, the basis functions are very narrow (ill-conditioned if centers are close).
# If sigma is too large, they are very broad (less expressive, similar to low-order polynomial).
# Let's estimate a good sigma. The distance between centers is approx (6.0 - 0.1) / 9 = 0.65.
# A sigma of 0.5 or 0.6 seems reasonable for overlap.
# Let's try sigma = 0.6. Then alpha = 1 / (2 * sigma^2).
SIGMA = 0.6
ALPHA = 1.0 / (2.0 * SIGMA**2)

# Precompute nothing, as x varies. But we can vectorize the calculation.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed-center Gaussian RBF basis.
    params: array of 10 weights.
    """
    if len(params) != 10:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=float)
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (10,)
    # We want a matrix R of shape (N, 10) where R[i, j] = exp(-ALPHA * (x[i] - centers[j])^2)
    
    # Vectorized computation
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    R = np.exp(-ALPHA * diff**2)
    
    # Prediction is R @ params
    pred = R @ params
    return pred

# EVOLVE-BLOCK-START
# No additional logic needed. The evaluator's BFGS will fit the 10 weights.
# The linear nature ensures convexity.
# The choice of RBF vs B-Spline is the structural change.
# EVOLVE-BLOCK-END
