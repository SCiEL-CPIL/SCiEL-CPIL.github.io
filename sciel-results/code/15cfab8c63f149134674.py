# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    
    # Strategy: 3 Gaussian Radial Basis Functions (RBFs)
    # Parameters:
    # p[0], p[1], p[2] : Centers (mu) for 3 RBFs
    # p[3], p[4], p[5] : Log-widths (log_sigma) for 3 RBFs
    # p[6], p[7], p[8] : Amplitudes (amp) for 3 RBFs
    # p[9]             : Global bias/offset
    
    # This uses all 10 parameters to define a sum of 3 Gaussians plus a constant.
    # y = sum_{i=1 to 3} amp_i * exp(-0.5 * ((x - mu_i) / sigma_i)^2) + bias
    
    # This is highly flexible for local features.
    # BFGS will optimize all parameters.
    # Starting from all ones:
    # mu = 1, log_sigma = 1 => sigma = e ~ 2.718
    # amp = 1
    # bias = 1
    # Initial guess: 3 bumps centered at 1, width 2.7, height 1, plus 1.
    # Total height ~ 4.
    
    mu1 = p[0]
    mu2 = p[1]
    mu3 = p[2]
    
    # Use exp for sigma to ensure positivity.
    # If p is large, sigma is large (broad). If p is small/negative, sigma is small (sharp).
    # exp(1) ~ 2.7.
    # To avoid overflow in exp if p is very large (e.g., > 700), we clip p.
    # However, BFGS usually keeps params in reasonable ranges.
    # Let's clip log_sigma to [-10, 10] to be safe.
    log_sig1 = np.clip(p[3], -10, 10)
    log_sig2 = np.clip(p[4], -10, 10)
    log_sig3 = np.clip(p[5], -10, 10)
    
    sig1 = np.exp(log_sig1)
    sig2 = np.exp(log_sig2)
    sig3 = np.exp(log_sig3)
    
    amp1 = p[6]
    amp2 = p[7]
    amp3 = p[8]
    
    bias = p[9]
    
    # Compute RBFs
    # To ensure numerical stability, we calculate the exponent carefully.
    # exp(-0.5 * z^2) where z = (x-mu)/sig
    
    def safe_rbf(x, mu, sig, amp):
        z = (x - mu) / sig
        # Clip z to prevent overflow in exp if sig is tiny and x-mu is large?
        # exp(-0.5 * z^2). If z is 1000, exp(-500000) is 0.
        # If z is 0, exp(0) is 1.
        # The only issue is if sig is 0, but exp(log_sig) is never 0.
        # So this is safe.
        return amp * np.exp(-0.5 * z * z)
    
    rbf1 = safe_rbf(x, mu1, sig1, amp1)
    rbf2 = safe_rbf(x, mu2, sig2, amp2)
    rbf3 = safe_rbf(x, mu3, sig3, amp3)
    
    result = rbf1 + rbf2 + rbf3 + bias
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to a simple linear combination of first two params
        return p[0] * x + p[1]
    
    return result
# EVOLVE-BLOCK-END
