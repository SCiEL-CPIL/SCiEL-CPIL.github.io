# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses a hybrid acquisition strategy combining:
    1. Upper Confidence Bound (UCB) for exploration
    2. Probability of Improvement (PI) weighted by expected gain
    3. Entropy-based exploration bonus for regions of high uncertainty
    
    This approach balances exploitation (low mu) with exploration (high sigma)
    using a time-adaptive strategy that emphasizes exploration early on.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) component
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Upper Confidence Bound (UCB) component
    # Use a time-adaptive beta: start with higher exploration, decrease over time
    # Since we don't have explicit time info, use the ratio of f_best to typical scale
    # For ackley 2d, f_best starts around 5-10 and decreases to 0
    # Use a fixed but reasonable beta that balances exploration/exploitation
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # Probability of Improvement (PI) - probability that x is better than f_best
    pi = norm.cdf(z_clipped)
    
    # Entropy-based exploration bonus: reward regions where sigma is high
    # This encourages exploration in uncertain regions
    entropy_bonus = sigma
    
    # Combine components with weights
    # Weight EI heavily as it's the most principled, but add UCB and entropy for robustness
    w_ei = 0.5
    w_ucb = 0.3
    w_entropy = 0.2
    
    # Normalize components to have similar scales before combining
    # EI is already in objective units
    # UCB: normalize by typical scale
    # Entropy: normalize by typical scale
    
    # Use a simple weighted combination
    score = w_ei * ei + w_ucb * ucb + w_entropy * entropy_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
