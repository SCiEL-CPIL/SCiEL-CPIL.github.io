"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Quantile-based exploration strategy:
    - Select points in the top 5% of uncertainty (high sigma rank).
    - Only consider those whose predicted mean is below f_best (i.e., potentially better).
    - Score is negative predicted mean plus a penalty derived from empirical rank of uncertainty.
    
    Actually, re-reading the directive: "negative predicted mean plus a penalty derived from the empirical rank of the uncertainty among candidates, effectively selecting points that are in the top 5% of uncertainty but only if their mean is below the current best."
    
    Let me interpret this carefully:
    - We want higher scores to mean "evaluate next".
    - The strategy focuses on points with high uncertainty (top 5% of sigma) AND low mean (below f_best).
    - Score = -mu + penalty(sigma_rank)
    
    But this seems odd because -mu would favor very low means, and the penalty from rank would... 
    Let me think about this differently. 
    
    Actually, a common quantile-based approach:
    1. Compute the rank of each candidate's sigma among all candidates.
    2. Only keep candidates whose sigma is in the top 5% (high uncertainty).
    3. Among those, only keep candidates whose mu < f_best.
    4. Score them based on how promising they are.
    
    But the directive says "negative predicted mean plus a penalty derived from the empirical rank of the uncertainty". 
    If penalty is positive and added to -mu, then higher uncertainty rank (higher penalty) increases score, which makes sense for exploration.
    
    Let's say:
    - rank = percentile rank of sigma (0 to 1, where 1 is highest sigma)
    - Only consider points where rank >= 0.95 (top 5% uncertainty) AND mu < f_best
    - For those points, score = -mu + alpha * rank
    - For others, score = -inf or a very low value
    
    But we need to return finite scores for all 2000 points. So for points not meeting the criteria, we can give them a very low score (e.g., -1e10) but still finite.
    
    Let me implement this.
    """
    # Compute percentile ranks of sigma (0 to 1, higher = more uncertain)
    sigma_ranks = np.argsort(np.argsort(sigma)) / (len(sigma) - 1)
    
    # Top 5% of uncertainty
    top_5_pct = sigma_ranks >= 0.95
    
    # Mean below current best (potentially better)
    below_best = mu < f_best
    
    # Valid candidates: top 5% uncertainty AND mean below best
    valid = top_5_pct & below_best
    
    # Score: -mu + penalty from rank
    # Use a moderate alpha to balance exploitation vs exploration
    alpha = 1.0
    scores = np.full(len(mu), -1e10)  # Very low default
    scores[valid] = -mu[valid] + alpha * sigma_ranks[valid]
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
