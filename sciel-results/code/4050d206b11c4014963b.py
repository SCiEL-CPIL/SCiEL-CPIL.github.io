"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-12

def acquisition(mu, sigma, f_best, cost):
    """
    Expected Regret per unit cost acquisition function.
    
    ER(x) = E[max(f_best - f(x), 0)]
    For a Gaussian f(x) ~ N(mu, sigma^2):
    ER = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    This is mathematically identical to EI, but we frame it as expected regret
    and normalize by cost. The key structural difference from the seed is that
    we use a more robust formulation and add small numerical safeguards.
    
    Actually, EI and ER are the same formula. Let's think of a truly different
    structural change.
    
    Alternative: Use Probability of Improvement (PI) weighted by uncertainty,
    or use a hybrid that balances exploration and exploitation differently.
    
    Let's try: UCB with cost normalization.
    UCB(x) = mu(x) - beta * sigma(x)  (since we minimize, we want low mu and high sigma)
    But UCB doesn't naturally account for f_best.
    
    Better: Let's use a modified EI that accounts for the multimodal nature of Rastrigin.
    Rastrigin has many local minima. The GP may have high uncertainty in regions far
    from observed points. We should encourage exploration of unexplored regions.
    
    New approach: Use Expected Improvement but with an exploration bonus based on
    the distance to the nearest observed point (proximity to data). However, we
    don't have access to the observed points directly, only mu and sigma.
    
    Actually, sigma from the GP already encodes uncertainty about the function value,
    which is related to distance from observed points. So EI already captures this.
    
    Let's try a different structural change: Use a quantile-based acquisition.
    Instead of maximizing expected improvement, maximize the probability that
    f(x) < f_best - delta for some delta, or use the lower quantile of the
    predictive distribution.
    
    New idea: Use the expected minimum of the predictive distribution.
    E[min(f(x), f_best)] = ... this is complex.
    
    Simpler structural change: Use a cost-aware version of Probability of Improvement
    combined with Expected Improvement, or use a different normalization.
    
    Let's try: Normalize EI by (cost * (1 + sigma)) to penalize high-cost, high-uncertainty
    points more, encouraging the algorithm to find improvements in cheaper regions first.
    
    Actually, let's go with a cleaner structural change: Use Expected Hypervolume Improvement
    (EHVI) approximation, or use a simple but different acquisition: 
    score = (f_best - mu) / (cost * (sigma + eps))
    This is the "probability of improvement" style but scaled differently.
    
    Let me think about what makes Rastrigin hard: many local minima. The GP needs to
    explore broadly. Pure EI can get stuck. A more exploratory acquisition might help.
    
    Let's use: score = (f_best - mu) / cost - lambda * log(1 + sigma)
    This adds an exploration bonus.
    
    Or simpler: Use UCB with cost:
    score = (mu - beta * sigma) / cost  -- no, this favors low mu which is good for minimization
    but we want to find points that might be better than f_best.
    
    For minimization, UCB would be: score = mu - beta * sigma (we want this to be low,
    but we return higher = better, so score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost)
    
    Let's try a hybrid: 0.5 * EI/cost + 0.5 * (sigma / cost)
    This balances exploitation (EI) and exploration (sigma).
    """
    # Compute z-score
    z = (f_best - mu) / (sigma + _EPS)
    
    # Expected Improvement (same as Expected Regret for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: use sigma directly as a measure of uncertainty
    # Combine EI with exploration term
    exploration = sigma
    
    # Weighted combination: 70% exploitation, 30% exploration
    score = (0.7 * ei + 0.3 * exploration) / (cost + _EPS)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
