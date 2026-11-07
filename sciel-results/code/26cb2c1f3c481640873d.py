"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Predictive Entropy Search (PES) variant acquisition function.
    
    Approximates the mutual information I(f(x_i); m*) where m* is the global
    minimum, by computing the expected KL divergence between the posterior
    distribution over the minimum before and after observing f(x_i).
    
    For each candidate x_i with posterior N(mu_i, sigma_i^2):
    1. Compute the prior probability that x_i is the new minimum:
       p_i = P(f(x_i) < f_best) = Phi((f_best - mu_i) / sigma_i)
    
    2. The entropy reduction is approximated by the information gain from
       observing f(x_i), which depends on:
       - How likely x_i is to improve on f_best (exploitation)
       - The uncertainty sigma_i (exploration - high uncertainty means
         more information potential)
       - The relative position in the landscape (f_best - mu_i)
    
    3. PES score combines:
       - Log-likelihood ratio: how surprising it would be to observe an
         improvement at x_i given the current posterior
       - Entropy of the predictive distribution at x_i
       - Weighted by the probability that x_i is competitive
    
    Practical approximation:
    score_i = log(1 + p_i * (1 + |z_i|)) * sigma_i * exp(-z_i^2 / (2 * max(1, n_eval)))
    
    where z_i = (f_best - mu_i) / sigma_i, and n_eval scales the exploration
    term based on how many evaluations have been done (approximated by the
    spread of mu values).
    
    This favors:
    - Points with moderate uncertainty (not too high, not too low)
    - Points that are close to but potentially below f_best
    - Points in regions where the GP is uncertain (high sigma)
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardize z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of improvement
    p_improve = norm.cdf(z_clipped)
    
    # Expected improvement (for reference)
    ei = (f_best - mu) * p_improve + sigma_safe * norm.pdf(z_clipped)
    
    # PES approximation: mutual information proxy
    # The information gain is proportional to:
    # 1. The entropy of the predictive distribution: H = 0.5 * log(2*pi*e*sigma^2)
    # 2. The probability that this point is competitive
    # 3. A term that rewards moderate uncertainty (not too exploratory, not too exploitative)
    
    # Entropy term (normalized)
    entropy_term = np.log(sigma_safe + eps)
    
    # Competition term: how likely is this point to be near the minimum
    # Use a Gaussian kernel centered at z=0 (where mu = f_best)
    # Points with z slightly negative (mu slightly above f_best) are most valuable
    competition_term = np.exp(-0.5 * z_clipped ** 2)
    
    # Balance exploration and exploitation
    # Early on (high uncertainty), explore more; later, exploit more
    # Approximate "progress" by looking at the spread of mu values
    mu_range = np.ptp(mu)
    if mu_range < eps:
        progress = 0.0
    else:
        # If mu values are tightly clustered, we're in exploitation mode
        progress = np.clip(1.0 - mu_range / (np.std(mu) + eps), 0, 1)
    
    # Combine terms
    # Weight: exploration (entropy) vs exploitation (competition)
    explore_weight = 0.3 + 0.7 * (1 - progress)
    exploit_weight = 1.0
    
    # PES score: expected information gain
    # Higher when:
    # - p_improve is moderate (not 0, not 1)
    # - sigma is moderate (high enough to have information, low enough to be relevant)
    # - z is near 0 (point is competitive)
    
    # Use a soft information gain metric
    # I ≈ E[log(p(f(x_i) < f_best | observed) / p(f(x_i) < f_best))]
    # Approximate by: p_improve * log(1 + sigma_safe * |z|) * competition
    
    # Simpler and more robust PES variant:
    # score = p_improve * (1 + log(sigma_safe + 1)) * exp(-z^2 / 4)
    # This rewards points that are likely to improve AND have useful uncertainty
    
    p_term = p_improve
    sigma_term = 1.0 + np.log1p(sigma_safe)  # log(1 + sigma), stable
    z_term = np.exp(-0.5 * z_clipped ** 2 / 2.0)  # Gaussian around z=0, wider
    
    score = p_term * sigma_term * z_term
    
    # Add a small exploration bonus for high-uncertainty regions
    # that are still potentially competitive
    exploration_bonus = explore_weight * sigma_safe * np.exp(-z_clipped ** 2 / 4.0) * 0.1
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Normalize to [0, 1] range for stability
    score_max = np.max(score)
    if score_max > eps:
        score = score / score_max
    
    return score
# EVOLVE-BLOCK-END
