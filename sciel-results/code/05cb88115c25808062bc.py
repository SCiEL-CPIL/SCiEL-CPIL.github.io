# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid approach that blends exploitation (Expected Improvement)
    with exploration (uncertainty) in a smooth, adaptive manner.
    The blend weight shifts from exploration to exploitation as f_best
    approaches the known minimum (0), and uncertainty is normalized to
    ensure the exploration term is on a comparable scale to EI.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)
    
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        return np.zeros(len(mu))
    
    # Ensure sigma is finite and non-negative
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    s = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (fb - mu) / s
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (fb - mu) * cdf_z + s * pdf_z
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Normalize sigma to [0, 1] range for blending
    s_max = np.max(s)
    if s_max > 1e-12:
        s_norm = s / s_max
    else:
        s_norm = s
    
    # Adaptive blend weight:
    # When f_best is far from 0 (minimum), we want more exploration
    # When f_best is close to 0, we want more exploitation
    # Use a smooth transition based on |f_best|
    # f_best starts around 25 (E[x^2] for uniform on [-5,5]) and decreases to ~0
    # Transition point: around f_best = 1.0
    # weight_exploit = sigmoid-like function of (1 - |f_best|)
    # At f_best = 0: weight_exploit ~ 1 (pure exploitation)
    # At f_best = 5: weight_exploit ~ 0.1 (mostly exploration)
    # At f_best = 1: weight_exploit ~ 0.5 (balanced)
    
    # Use a smooth sigmoid centered at f_best = 1.0
    # w = 1 / (1 + exp(-k * (c - |f_best|)))
    # At |f_best| = 0: w = 1/(1+exp(-k*c))
    # At |f_best| = c: w = 0.5
    # At |f_best| = large: w ~ 0
    
    k = 2.0  # steepness
    c = 1.0  # center point
    abs_fb = abs(fb)
    w = 1.0 / (1.0 + np.exp(-k * (c - abs_fb)))
    
    # Clamp w to [0.05, 0.95] to avoid extreme values
    w = np.clip(w, 0.05, 0.95)
    
    # Combine: weighted blend of exploitation and exploration
    # Higher score = better
    scores = w * ei_norm + (1.0 - w) * s_norm
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
