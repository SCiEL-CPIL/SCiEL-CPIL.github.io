import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Probability of Improvement (PoI) with a cost penalty.
    PoI is often more robust than EI in early stages and handles
    the trade-off between exploitation and exploration differently.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning high probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is Phi(z)
    poi = norm.cdf(z)
    
    # Cost-awareness:
    # The cost is 1 + mean(x) where x is scaled to [0, 1].
    # So cost is in [1, 2].
    # We want to penalize high cost.
    # A simple division (1/cost) is linear.
    # Let's try a slightly stronger penalty or a different normalization.
    # The current program uses EI / cost.
    # Let's stick to the idea of normalizing the acquisition by cost.
    
    # To make it more "cost-aware" in a way that might improve exploration
    # in cheap regions, we can use a power law or just stick to 1/cost.
    # However, the prompt mentions "cost-aware acquisition".
    # Let's try a slight modification: 
    # score = PoI * (1 / cost^alpha)
    # If alpha > 1, we penalize cost more heavily.
    # If alpha < 1, we are less sensitive to cost.
    # Given the budget is 30 and initial cost is ~1.5, and we have 2000 candidates,
    # we are likely evaluating many points.
    
    # Let's use a simple ratio but ensure numerical stability.
    # Also, PoI is bounded in [0, 1]. EI is unbounded (theoretically).
    # Normalizing by cost makes the scores comparable.
    
    scores = poi / cost
    
    # Optional: Add a tiny epsilon to avoid zeros if cost is high and poi is low?
    # No, 0 is a valid score.
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
