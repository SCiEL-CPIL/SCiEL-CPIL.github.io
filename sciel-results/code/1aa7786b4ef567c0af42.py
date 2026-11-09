import numpy as np
import time

# Pre-compute lookup tables for fast normal CDF and PDF
_Z_LUT_MIN = -12.0
_Z_LUT_MAX = 12.0
_Z_LUT_SIZE = 1_000_001

def _build_luts():
    z = np.linspace(_Z_LUT_MIN, _Z_LUT_MAX, _Z_LUT_SIZE)
    cdf = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return z, cdf, pdf

_z_lut, _cdf_lut, _pdf_lut = _build_luts()

def _fast_norm_cdf(z):
    """Vectorized fast normal CDF using lookup table with linear interpolation."""
    z = np.asarray(z)
    idx = (z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_SIZE - 1)
    idx = np.clip(idx, 0, _Z_LUT_SIZE - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, _Z_LUT_SIZE - 1)
    frac = idx - i0
    return _cdf_lut[i0] * (1 - frac) + _cdf_lut[i1] * frac

def _fast_norm_pdf(z):
    """Vectorized fast normal PDF using lookup table with linear interpolation."""
    z = np.asarray(z)
    idx = (z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_SIZE - 1)
    idx = np.clip(idx, 0, _Z_LUT_SIZE - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, _Z_LUT_SIZE - 1)
    frac = idx - i0
    return _pdf_lut[i0] * (1 - frac) + _pdf_lut[i1] * frac

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a multi-objective Pareto-front inspired approach:
    - Computes EI (expected improvement) for exploitation
    - Computes UCB-like score for exploration
    - Normalizes both to [0,1] relative to their own distributions
    - Combines with adaptive weighting based on convergence state
    - Divides by cost for cost-awareness
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    n = len(mu)
    
    # --- Compute Expected Improvement for minimization ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    # EI should be non-negative; clip tiny negatives from numerical error
    ei = np.maximum(ei, 0.0)
    
    # --- Compute UCB score for minimization ---
    # For minimization, UCB = mu - kappa * sigma (we want this to be low)
    # Score (higher = better) = -(mu - kappa * sigma) = kappa * sigma - mu
    # Adaptive kappa: more exploration when f_best is far from minimum
    base_kappa = 1.5
    kappa = base_kappa * (1.0 + f_best / (f_best + 1.0))
    ucb_score = kappa * sigma - mu
    
    # --- Normalize EI and UCB to [0, 1] relative to their distributions ---
    # This creates a probabilistic utility that's comparable across scales
    ei_max = np.max(ei)
    if ei_max > 1e-15:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-15:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # --- Adaptive weighting based on convergence state ---
    # When f_best is high (early), weight UCB (exploration) more
    # When f_best is low (late), weight EI (exploitation) more
    # Use a smooth transition
    weight_ucb = 0.3 + 0.5 * (f_best / (f_best + 0.5))
    weight_ucb = np.clip(weight_ucb, 0.0, 1.0)
    weight_ei = 1.0 - weight_ucb
    
    # --- Combine normalized scores ---
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # --- Cost-aware: divide by cost ---
    # Higher cost means less value per unit cost
    score = combined / cost
    
    # --- Add a small exploration bonus for high uncertainty regions ---
    # This helps escape local optima
    sigma_max = np.max(sigma)
    if sigma_max > 1e-15:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Small bonus: 5% weight on normalized sigma
    score += 0.05 * sigma_norm / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: if we're taking too long, return immediately
    # (This is a simple acquisition, so this should rarely trigger)
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        return score
    
    return score
