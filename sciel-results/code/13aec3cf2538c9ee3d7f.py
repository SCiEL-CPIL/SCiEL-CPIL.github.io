"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining adaptive LCB and EI for Bayesian optimization.
    
    Args:
        mu: array of shape (n,) - GP mean predictions
        sigma: array of shape (n,) - GP standard deviation predictions
        f_best: float - best observed value so far (lower is better)
    
    Returns:
        array of shape (n,) - acquisition scores (higher = more promising)
    """
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, eps)
    
    # Expected Improvement (EI) - higher is better for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) - lower is better, so negate for higher-is-better scoring
    # Adaptive beta: increase exploration when uncertainty is high relative to potential improvement
    sigma_max = np.max(sigma)
    mu_min = np.min(mu)
    
    # Potential improvement: how much better we could potentially do
    potential_improvement = f_best - mu_min
    
    # Adaptive beta: larger when potential improvement is large relative to uncertainty
    if sigma_max > eps:
        # Normalize potential improvement by typical uncertainty
        ratio = potential_improvement / (sigma_max + eps)
        # Beta ranges from ~1 (exploitative) to ~3 (exploratory)
        beta = 1.0 + 2.0 * np.tanh(ratio / 5.0)
    else:
        beta = 1.0
    
    lcb_score = -(mu - beta * sigma)  # Negate because lower LCB is better
    
    # Blend EI and LCB: use more LCB early (high uncertainty), more EI later
    # Use sigma_max as a proxy for how much we've learned
    # When sigma_max is large, we're early -> weight LCB more
    # When sigma_max is small, we're late -> weight EI more
    
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei)
    lcb_max = np.max(lcb_score)
    
    if ei_max > eps:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    if lcb_max > eps:
        lcb_norm = lcb_score / lcb_max
    else:
        lcb_norm = lcb_score
    
    # Weight: more LCB when uncertainty is high
    # sigma_max relative to the scale of the problem
    # Use a simple heuristic: weight_lcb = 0.5 + 0.5 * tanh(sigma_max)
    # But we need a scale reference. Use the range of mu as a proxy.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range > eps:
        # How much uncertainty relative to the range of predictions
        uncertainty_ratio = sigma_max / mu_range
    else:
        uncertainty_ratio = 1.0
    
    # Weight for LCB: higher when uncertainty is high
    w_lcb = 0.5 + 0.5 * np.tanh(uncertainty_ratio * 2.0)
    w_ei = 1.0 - w_lcb
    
    # Combined score
    score = w_lcb * lcb_norm + w_ei * ei_norm
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
