"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a structural blend of:
      - Expected Improvement (EI): targets exploitation near current best
      - Probability of Improvement (POI): smooth exploration signal
      - Inverse-variance weighted confidence: favors low-uncertainty regions
        where the GP is confident about low values
    Combined with a cost-aware penalty.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP standard deviation predictions
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        fallback = -mu / cost_safe + 0.5 * sigma_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # --- Component 1: Expected Improvement ---
    diff = f_best - mu
    z = diff / sigma_safe
    z_clipped = np.clip(z, -15, 15)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = diff * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # --- Component 2: Probability of Improvement (smooth exploration) ---
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = cdf  # Already computed above
    
    # --- Component 3: Confidence-weighted negative mean ---
    # In low-uncertainty regions, trust the mean more.
    # In high-uncertainty regions, add a moderate exploration bonus.
    # Use a softmax-like weighting: weight = 1 / (1 + (sigma / sigma_ref)^2)
    sigma_ref = 0.1  # Reference scale for uncertainty
    confidence_weight = 1.0 / (1.0 + (sigma_safe / sigma_ref) ** 2)
    
    # Negative mean scaled by confidence: high confidence + low mean = high score
    conf_neg_mu = -mu * confidence_weight
    
    # Exploration bonus in uncertain regions: sigma * (1 - confidence_weight)
    exploration_bonus = sigma_safe * (1.0 - confidence_weight) * 0.5
    
    # --- Combine components ---
    # Normalize each component to [0, 1] range for balanced blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # POI is already in [0, 1]
    
    conf_neg_mu_max = np.max(conf_neg_mu)
    conf_neg_mu_min = np.min(conf_neg_mu)
    conf_range = conf_neg_mu_max - conf_neg_mu_min
    if conf_range > 1e-12:
        conf_norm = (conf_neg_mu - conf_neg_mu_min) / conf_range
    else:
        conf_norm = np.zeros_like(mu)
    
    # Blend: 50% EI, 25% POI, 25% confidence-weighted mean
    blended = 0.50 * ei_norm + 0.25 * poi + 0.25 * conf_norm
    
    # Add exploration bonus
    blended = blended + exploration_bonus / (np.max(exploration_bonus) + 1e-12)
    
    # --- Cost-aware penalty ---
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    # Use a moderate cost penalty: divide by cost^0.5 (softer than linear)
    cost_penalty = np.power(cost_safe, 0.5)
    
    score = blended / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Add a small constant to ensure positive scores for ranking stability
    score = score + 1e-6
    
    return score
# EVOLVE-BLOCK-END
