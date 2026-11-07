# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a multiplicative feasibility gate.
    
    The score is Expected Improvement (EI) only if the predicted cost is below
    a dynamically determined threshold (current average cost per unit improvement
    proxy). This creates a hard budget-aware gate that forces exploitation of
    low-cost regions until GP uncertainty there is exhausted.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: use a multiplicative gate based on cost.
    # The idea: only allow evaluation if cost is "affordable" relative to the
    # expected improvement. We define a threshold as a fraction of the minimum
    # observed cost plus a small epsilon, creating a hard gate.
    
    # Compute a dynamic threshold: points with cost below this threshold are
    # considered "affordable". We use the 25th percentile of costs as a
    # dynamic threshold that adapts to the candidate distribution.
    cost_threshold = np.percentile(cost, 25) + 1e-9
    
    # Gate: only pass EI if cost is below threshold; otherwise heavily penalize
    # using a multiplicative factor that goes to 0 for expensive points.
    # This creates a hard feasibility constraint.
    affordable = cost < cost_threshold
    
    # For affordable points, return EI. For unaffordable points, return a
    # small value scaled by how far above the threshold the cost is.
    # Use a smooth but steep transition to avoid discontinuities.
    # ratio = cost / cost_threshold; for ratio > 1, score drops off rapidly.
    ratio = cost / (cost_threshold + 1e-9)
    
    # Multiplicative decay: for ratio <= 1, factor = 1.0; for ratio > 1,
    # factor decays as exp(-(ratio - 1)^2 * 10) to create a soft but strong gate.
    decay = np.where(
        ratio <= 1.0,
        1.0,
        np.exp(-10.0 * (ratio - 1.0) ** 2)
    )
    
    scores = ei * decay
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
