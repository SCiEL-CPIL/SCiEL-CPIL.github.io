"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Sum of 3 Gaussian bumps + constant offset.
    This allows adaptive localization of features.
    params[0:3]: centers (mu)
    params[3:6]: widths (sigma)
    params[6:9]: amplitudes (A)
    params[9]: offset (C)
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)

    mu = params[0:3]
    sigma = np.abs(params[3:6]) + 1e-6  # Ensure positive widths
    A = params[6:9]
    C = params[9]

    # Calculate sum of Gaussian contributions
    # Using broadcasting: x is (N,), mu is (3,) -> (N, 3)
    diff = x[:, np.newaxis] - mu[np.newaxis, :]
    
    # Gaussian term: exp(-0.5 * (diff / sigma)^2)
    # We can vectorize this: sigma is (3,), so diff^2 / sigma^2 is (N, 3)
    sigma_sq = sigma * sigma
    exponent = -0.5 * (diff * diff) / sigma_sq
    
    # To prevent overflow in exp (though diff/sigma is usually small, 
    # if sigma is tiny and diff is large, exp(0) = 1, which is fine.
    # If sigma is huge, exponent is ~0, exp(0)=1.
    # The main risk is numerical instability if sigma is extremely small.
    # Clipping exponent helps but isn't strictly necessary if sigma is bounded away from 0.
    # However, if sigma -> 0, the Gaussian becomes a spike.
    # Let's just compute the exponentials.
    gaussians = np.exp(exponent)
    
    # Weighted sum
    pred = np.sum(A * gaussians, axis=1) + C

    return pred
# EVOLVE-BLOCK-END
