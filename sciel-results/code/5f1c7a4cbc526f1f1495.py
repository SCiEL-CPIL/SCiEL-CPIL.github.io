import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Distance-Weighted Expected Improvement with Thompson Sampling.
    
    Combines the deterministic exploitation of Expected Improvement (EI) with a
    "Distance-to-Best" repulsive force. This force scales the EI score by a factor
    that increases with posterior uncertainty (sigma), acting as a proxy for distance
    from the current best point. This encourages the search to jump to distant,
    unexplored basins, which is critical for escaping the dense local minima of the
    Michalewicz function. A Thompson Sampling term adds stochastic robustness.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Component 1: Expected Improvement (Deterministic Exploitation) ---
    # EI measures the expected amount by which a candidate improves upon f_best.
    # Standardized improvement
    z_ei = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # --- Component 2: Distance-to-Best Repulsive Force ---
    # We use the posterior uncertainty (sigma) as a proxy for distance from the
    # current best point. Points with high sigma are likely in unexplored regions
    # (different basins). We weight EI by a factor that increases with sigma to
    # encourage exploration of distant regions.
    
    # Compute a reference scale for uncertainty (median of sigma across candidates)
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-9:
        sigma_median = 1e-9
    
    # Normalized uncertainty
    sigma_norm = sigma_safe / sigma_median
    
    # Weight: Points with higher uncertainty get a higher weight.
    # This acts as a repulsive force from the low-uncertainty (exploited) regions.
    # The coefficient 0.5 balances exploration and exploitation.
    weight = 1.0 + 0.5 * sigma_norm
    
    weighted_ei = weight * ei
    
    # --- Component 3: Thompson Sampling (Stochastic Robustness) ---
    # Add a small TS component to avoid getting stuck in a single deterministic path.
    # This helps with "unlucky" deterministic choices and multimodal landscapes.
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    ts_score = -y_samples
    
    # Normalize TS score to be comparable to EI?
    # EI is in objective units. TS is in objective units.
    # But EI is usually smaller than the full function value range.
    # Let's scale TS down to be a secondary term.
    ts_weight = 0.1
    ts_contribution = ts_weight * ts_score
    
    # Combine all components
    scores = weighted_ei + ts_contribution
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline
    if time.time() > deadline:
        # If we're over deadline, just return what we have (shouldn't happen for this simple computation)
        pass
    
    return scores
