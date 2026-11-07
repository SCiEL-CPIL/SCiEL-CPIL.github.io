"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Use a hybrid of Expected Improvement and Upper Confidence Bound
    # to balance exploitation and exploration for rastrigin's local minima.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    # Using xi=0 for standard EI, which focuses on finding the global minimum
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Compute Upper Confidence Bound (UCB)
    # The beta parameter controls the trade-off between exploitation (mu) and exploration (sigma).
    # A higher beta encourages more exploration, which is helpful for escaping local minima in rastrigin.
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB.
    # EI is a measure of expected gain, while UCB is a measure of potential gain with uncertainty.
    # We take the maximum to ensure we don't miss out on either a high-certainty improvement or a high-potential exploration.
    score = np.maximum(ei, -ucb)
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    # Adding a small constant to cost prevents division by zero and ensures smoothness
    cost_safe = cost + 1e-8
    score = score / cost_safe
    
    # Add a small exploration bonus based on sigma to further encourage exploring uncertain regions
    # This helps in escaping local minima by slightly favoring regions with higher uncertainty
    exploration_bonus = 0.1 * sigma_safe / (1 + sigma_safe)
    score += exploration_bonus
    
    return score
# EVOLVE-BLOCK-END
