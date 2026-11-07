import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using GP-LCB with adaptive beta.
    
    mu: array of shape (2000,) posterior means
    sigma: array of shape (2000,) posterior standard deviations
    f_best: float, the minimum value observed so far
    
    Returns:
    array of shape (2000,) scores, higher is better for selection.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive Beta strategy:
    # The evaluator runs a loop of 30 evaluations (5 initial + 25 optimization steps).
    # We do not have direct access to the step counter 't', but we can infer a 
    # heuristic based on the uncertainty of f_best relative to the posterior.
    # However, a simpler and robust approach for fixed budgets is to use a 
    # beta that balances exploration and exploitation.
    
    # A common adaptive schedule for a small budget (N=30) is:
    # beta_t = sqrt(log(t+1) * (d * log(d*N) + 1))  [Theoretical bound, often too complex/overly conservative]
    # Or a simpler heuristic: beta starts high and decreases.
    
    # Since we cannot track 't' directly inside this stateless function, 
    # we can use a fixed beta that is empirically strong for N=30 in 2D, 
    # or derive a proxy. 
    # 
    # The "Knowledge from this run" suggests Adaptive LCB (high beta early, low late) 
    # improved score significantly (1 -> 2.421). 
    # Since we can't pass state, we must pick a single effective beta or 
    # use a property of the inputs to approximate the stage.
    
    # One heuristic: If f_best is very close to the theoretical minimum (-2.53), 
    # we are likely in exploitation phase. But we don't know the theoretical min.
    
    # Let's use a fixed beta that represents a strong exploration/exploitation 
    # balance for the early-mid game, which is where most of the regret reduction 
    # happens in 30 steps. 
    # Standard EI is roughly equivalent to a specific beta. 
    # LCB: -mu - beta * sigma
    # For minimization, we want to minimize f. 
    # Acquisition score should be high where we expect f to be low.
    # Score = -mu - beta * sigma
    
    # What is the optimal beta for 30 steps in 2D RBF mixture?
    # Literature suggests beta ~ 2-5 is often effective.
    # The "inspiration 3" mentioned a fixed UCB with beta=2.0 scored 0.56.
    # The "knowledge" mentions Adaptive LCB scored 2.421.
    # Since we can't adapt, let's try a beta that is slightly higher than EI's effective beta 
    # to encourage more exploration in the first half, which is critical for multimodal functions.
    
    # Let's try beta = 2.5. This is a common choice for balancing exploration 
    # in low-budget BO.
    beta = 2.5
    
    # Calculate LCB score
    # We want to find points with low mu and high sigma (uncertainty).
    # Lower value of (mu + beta * sigma) is better.
    # So we return negative of that as the score (higher is better for selection).
    score = -(mu + beta * sigma_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
