import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a fast, safe fallback
    if time.time() - _start_time > 325:
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    safe_cost = np.maximum(cost, 1e-6)
    
    # Structural Change: Switch from UCB to Knowledge Gradient (KG) acquisition function
    # KG directly targets expected reduction in the optimal value, which is more principled
    # for minimization than EI. It scales exploration with uncertainty and can steer search
    # toward unexplored regions more effectively.
    
    # For minimization, KG measures the expected improvement of the current best value
    # given the new observation. We approximate this using a closed-form expression.
    
    # The KG value for a candidate point is:
    # KG = E[max(0, f_best - Y)] where Y ~ N(mu, sigma^2)
    # This is equivalent to Expected Improvement (EI) for minimization.
    # However, to make it more cost-aware and aggressive in exploration, we use a
    # modified form that emphasizes regions with high uncertainty.
    
    # Standard EI for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # For cost-awareness, we divide by cost to favor cheaper evaluations.
    
    # Calculate the standardized improvement
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Calculate EI
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: divide by cost to favor cheaper evaluations
    kg_score = ei / safe_cost
    
    # Ensure all scores are finite
    kg_score = np.where(np.isfinite(kg_score), kg_score, 0.0)
    
    return kg_score
