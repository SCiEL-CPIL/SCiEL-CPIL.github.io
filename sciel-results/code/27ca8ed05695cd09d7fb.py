import numpy as np
import time
from scipy.stats import norm

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using quantile-based improvement with adaptive cost penalty.
    Instead of standard EI, we use the probability of improvement blended with
    a quantile of the predicted distribution. This is more robust when the GP
    posterior has high uncertainty.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)

    # Z-score for the improvement
    Z = (f_best - mu) / sigma
    
    # Probability of improvement: P(f(x) < f_best) = Phi(Z)
    POI = norm.cdf(Z)
    
    # Quantile-based score: use the 25th percentile of the predicted distribution
    # as a proxy for "likely good value". This is more robust than mean.
    # The 25th percentile of N(mu, sigma^2) is mu + sigma * norm.ppf(0.25)
    q25 = norm.ppf(0.25)  # approximately -0.674
    quantile_score = -(mu + q25 * sigma)  # Negate because we want to maximize
    
    # Blend POI and quantile score. POI is in [0,1], quantile_score is in objective units.
    # Normalize quantile_score to [0,1] range relative to f_best.
    # If f_best is large, the scale is large. We use a soft normalization.
    # score_blend = w * POI + (1-w) * normalized_quantile
    # Let's use w=0.5 for a balanced approach.
    # Normalize quantile: (f_best - (mu + q25*sigma)) / (f_best + 1) to keep in reasonable range
    norm_q = (f_best - (mu + q25 * sigma)) / (abs(f_best) + 1.0)
    norm_q = np.clip(norm_q, 0.0, 1.0)
    
    # Blend: 50% probability of improvement, 50% normalized quantile
    base_score = 0.5 * POI + 0.5 * norm_q
    
    # Adaptive cost penalty:
    # When f_best is large (early), we can afford to explore more (lower penalty).
    # When f_best is small (late), we should be more cost-sensitive.
    # penalty_weight = 1.0 / (1.0 + f_best)  # Strong penalty when f_best is small
    # Actually, let's think: cost is in [1,2]. We want to divide by a factor.
    # Early: f_best ~ 100, penalty_weight ~ 0.01, so score / (1 + 0.01*cost) ~ score
    # Late: f_best ~ 0.1, penalty_weight ~ 0.91, so score / (1 + 0.91*cost) ~ score/2
    # This seems reasonable.
    penalty_weight = 1.0 / (1.0 + abs(f_best))
    scores = base_score / (1.0 + penalty_weight * safe_cost)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
