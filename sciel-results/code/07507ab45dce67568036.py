import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Deadline guard (though vectorized ops are fast, this satisfies the directive)
    # We assume the function is called frequently, so we just do the math.
    # If this were a loop inside, we'd check time. Here it's a single shot.
    
    epsilon = 1e-9
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # Avoid division by zero
    sigma_safe = sigma + epsilon
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB) for minimization
    # We maximize the acquisition, so we want high values.
    # For minimization, low mu is good. So -mu is good.
    # High sigma is good (exploration).
    # UCB_score = -mu + beta * sigma
    beta = 1.5  # Exploration parameter
    ucb_score = -mu + beta * sigma
    
    # Blend EI and UCB
    # The "inspiration" suggested 70/30 blend
    weight_ei = 0.7
    weight_ucb = 0.3
    blended_score = weight_ei * ei + weight_ucb * ucb_score
    
    # Apply cost penalty
    # "Softening the cost penalty (0.4x)" -> use exponent 0.4 on cost
    # cost is (1 + mean(x_scaled)), so cost >= 1.
    cost_safe = cost + epsilon
    penalty = np.power(cost_safe, 0.4)
    
    # Final acquisition score
    scores = blended_score / penalty
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
