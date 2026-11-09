"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s. We return safely before that.
_START_TIME = time.time()
DEADLINE = _START_TIME + 320.0  # 10s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    We use a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    normalized by cost. UCB with a moderate beta encourages exploration in early
    stages (critical for multimodal Griewank), while EI provides exploitation.
    
    We blend: score = (alpha * EI + (1 - alpha) * UCB) / cost
    where alpha adapts based on the current uncertainty level.
    
    Parameters:
        mu: mean of GP predictions, shape (2000,)
        sigma: std of GP predictions, shape (2000,)
        f_best: best observed value (scalar)
        cost: cost of evaluating each candidate, shape (2000,)
    
    Returns:
        2000 finite scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > DEADLINE:
        # Return a simple fallback: just EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB: mu + beta * sigma (we want to minimize, so we use -mu + beta*sigma
    # but since we're maximizing the acquisition score, we think of it as
    # potential improvement. For minimization, UCB-like score = -mu + beta*sigma
    # but EI already handles the sign. Let's use a different formulation.
    # 
    # Actually, for minimization:
    # - EI = E[f_best - f(x)] = (f_best - mu)*Phi(z) + sigma*phi(z), z=(f_best-mu)/sigma
    # - UCB for minimization: we want low mu and high sigma (uncertainty)
    #   A common approach: score_ucb = -mu + beta * sigma
    #   But we need to combine with EI. 
    #
    # Let's use a weighted combination:
    # score = w * (EI / cost) + (1-w) * (UCB_score / cost)
    # where UCB_score = -mu + beta * sigma (higher is better for exploration)
    #
    # However, -mu + beta*sigma can be negative or positive, while EI is non-negative.
    # To make them comparable, we can normalize or use a different blend.
    #
    # Alternative: Use EI with a temperature parameter, or use a q-EI variant.
    #
    # Let's try: score = (EI + beta * sigma) / cost
    # This adds an exploration bonus proportional to sigma.
    
    # Adaptive beta: higher when we have few observations (early stage), lower later.
    # We don't know the exact number of observations, but we can infer from cost.
    # Initial 5 points cost ~ 5 * (1 + mean) ≈ 5 * 1.5 = 7.5
    # Total budget is 30, so we have ~ 30/1.5 = 20 evaluations total.
    # 
    # Let's use a fixed moderate beta that balances exploration and exploitation.
    # From the knowledge: UCB with beta=1 improved score significantly.
    
    beta = 1.0
    
    # Combine EI and exploration bonus
    score = (ei + beta * sigma_safe) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
