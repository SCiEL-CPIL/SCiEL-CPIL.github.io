# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as t_dist
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines expected improvement with a regret-risk
    component using a Student-t approximation of the posterior.
    
    The regret risk is the probability that the true value exceeds f_best,
    which we approximate using a t-distribution over the posterior mean
    and variance. We maximize a weighted combination of EI and this probability.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean in objective units.
    sigma : array-like, shape (2000,)
        Posterior standard deviation in objective units.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Regret risk: probability that true value > f_best
    # Using t-distribution with df=3 (heavy tails for uncertainty)
    df = 3.0
    # Standardize: how many sigma away is f_best from mu
    t_stat = (f_best - mu) / sigma_safe
    # P(true value > f_best) = P(t < -t_stat) for the t-distribution
    # Actually, if Y ~ N(mu, sigma^2), P(Y > f_best) = P(Z > (f_best - mu)/sigma)
    # With t-approximation: P(Y > f_best) ≈ 1 - t.cdf(t_stat, df)
    regret_risk = 1.0 - t_dist.cdf(t_stat, df)
    
    # Combine: we want high EI and low regret risk
    # But the directive says to maximize the probability of improvement,
    # so let's weight the regret risk component.
    # Actually, re-reading: "maximize this probability to explicitly target 
    # high-improvement regions" — this seems contradictory since regret_risk 
    # is P(value > f_best), which we want to MINIMIZE.
    
    # Let me re-interpret: The "regret risk" is the probability of NOT improving,
    # so we want to minimize it, or equivalently maximize P(value < f_best).
    # P(value < f_best) = t_dist.cdf(t_stat, df)
    
    prob_improve = t_dist.cdf(t_stat, df)
    
    # Normalize EI to be comparable to probability
    # EI can be large, so let's scale it
    ei_scale = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ei_normalized = ei / ei_scale
    
    # Weighted combination
    weight_ei = 0.7
    weight_prob = 0.3
    
    scores = weight_ei * ei_normalized + weight_prob * prob_improve
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
