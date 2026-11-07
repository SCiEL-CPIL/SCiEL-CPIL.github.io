"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    EI drives exploitation towards known good areas.
    UCB drives exploration towards uncertain areas.
    The cost penalty is softened (power < 1) to allow exploration of 
    higher-cost regions if the expected gain is sufficient.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    epsilon = 1e-9
    sigma_safe = np.maximum(sigma, epsilon)
    cost_safe = np.maximum(cost, epsilon)
    
    # 1. Expected Improvement (EI) for minimization
    # We want to minimize, so improvement is f_best - f(x).
    # z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want low mu (good) and high sigma (uncertain).
    # Score = -mu + beta * sigma
    # Beta controls the exploration/exploitation trade-off.
    # A value around 1.0 to 2.0 is typical.
    beta = 1.5
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Blend EI and UCB
    # EI is in objective units (improvement).
    # UCB is in objective units (value).
    # They are comparable in scale, but EI is 0 when no improvement is expected.
    # We weight them to balance the two.
    weight_ei = 0.7
    weight_ucb = 0.3
    
    # To ensure the scores are positive and well-behaved, we can add a small offset
    # or just rely on the fact that EI is positive and UCB can be negative.
    # If UCB is very negative, it hurts the score. This is correct: if mu is very high,
    # we don't want to go there unless sigma is huge.
    
    blended_score = weight_ei * ei + weight_ucb * ucb_score
    
    # 4. Cost-aware penalty
    # Cost is 1 + mean(x_scaled). Min cost is 1.
    # Using a power less than 1 softens the penalty, allowing higher cost points
    # to be considered if they have high acquisition value.
    cost_penalty = np.power(cost_safe, 0.4)
    
    scores = blended_score / cost_penalty
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Ensure scores are non-negative? 
    # The task says "higher = evaluate next". It doesn't require positive scores.
    # However, if all scores are negative, the "best" (max) will be the least negative.
    # This is fine.
    
    return scores
# EVOLVE-BLOCK-END
