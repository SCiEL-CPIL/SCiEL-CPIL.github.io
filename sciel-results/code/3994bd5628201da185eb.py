import numpy as np
import time
from scipy.stats import norm, t as scipy_t

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a robust probabilistic regret minimization
    approach. Combines the structural idea of EI per unit cost with a Student-t
    distribution to account for model uncertainty, plus an adaptive exploration
    term that balances exploitation and exploration based on the relative
    uncertainty of each candidate.
    
    The key structural change from the baseline is the use of a Student-t based
    improvement metric, which is more robust to outliers in the improvement
    distribution, combined with an adaptive exploration weight that increases
    when uncertainty is high relative to the expected improvement.
    """
    # Track time for safety margin
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    # Standard Normal CDF and PDF
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Student-t based robust improvement metric
    # Use degrees of freedom that adapt to the scale of uncertainty
    # Lower DOF = heavier tails = more robust to outliers in the improvement
    # We use a fixed moderate DOF (5) for robustness while maintaining sensitivity
    dof = 5.0
    
    # Student-t CDF and PDF at z
    t_cdf_val = scipy_t.cdf(z_clipped, dof)
    t_pdf_val = scipy_t.pdf(z_clipped, dof)
    
    # Scale factor for Student-t: variance of t_dof is dof/(dof-2) for dof > 2
    t_scale = np.sqrt(dof / (dof - 2.0))
    
    # Student-t Expected Improvement (robust version)
    st_ei = (f_best - mu) * t_cdf_val + sigma * t_scale * t_pdf_val
    
    # Cost-aware scores
    score_ei = ei / cost_safe
    score_st = st_ei / cost_safe
    
    # Adaptive exploration term: when uncertainty is high relative to improvement,
    # we want to explore more. The exploration weight is proportional to the
    # ratio of sigma to the expected improvement magnitude.
    improvement_magnitude = np.maximum(f_best - mu, 1e-10)
    uncertainty_ratio = sigma_safe / improvement_magnitude
    
    # Clip uncertainty ratio to avoid extreme values
    uncertainty_ratio_clipped = np.clip(uncertainty_ratio, 0, 5)
    
    # Exploration score: sigma / cost, weighted by uncertainty ratio
    score_explore = (sigma_safe / cost_safe) * (1.0 + 0.5 * uncertainty_ratio_clipped)
    
    # Adaptive combination: use more robust metrics when uncertainty is high
    # Normalize sigma to [0, 1] for adaptive weighting
    sigma_max = np.max(sigma_safe) + 1e-10
    sigma_norm = sigma_safe / sigma_max
    
    # When sigma is high, trust the robust Student-t metric and exploration more
    # When sigma is low, the standard EI is more reliable
    w_st = 0.2 + 0.3 * sigma_norm  # Weight for Student-t: 0.2 to 0.5
    w_ei = 0.5 - 0.2 * sigma_norm  # Weight for standard EI: 0.5 to 0.3
    w_explore = 0.3 + 0.2 * sigma_norm  # Weight for exploration: 0.3 to 0.5
    
    # Normalize weights to sum to 1
    weight_sum = w_st + w_ei + w_explore
    w_st /= weight_sum
    w_ei /= weight_sum
    w_explore /= weight_sum
    
    scores = w_ei * score_ei + w_st * score_st + w_explore * score_explore
    
    # Check deadline
    if time.time() > deadline:
        # Return the simplest and fastest score as fallback
        return score_ei
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
