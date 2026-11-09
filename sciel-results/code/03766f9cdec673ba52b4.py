"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a model-and-optimize approach.
    
    1. Defines a family of cost-aware acquisition functions parameterized by beta.
       Score(x; beta) = EI(x) / cost(x) + beta * sigma(x) / cost(x)
    2. Optimizes beta by selecting the value that maximizes the Expected Improvement
       (EI) of the best candidate identified by that variant.
    3. Returns the scores for the optimal beta.
    """
    # Deadline guard: if we are close to the limit, return a safe, fast result
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Precompute terms that are independent of beta
    z = (f_best - mu) / sigma_safe
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Base cost-aware EI term
    base_score = ei / cost_safe
    
    # Exploration term component: sigma / cost
    expl_component = sigma_safe / cost_safe
    
    # Define a grid of beta values to test
    # Beta controls the balance between exploitation (EI) and exploration (sigma)
    # We use a logarithmic spacing to cover a wide range of exploration levels
    betas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    
    best_beta = 1.0
    best_ei_of_best = -np.inf
    best_scores = None
    
    # Iterate over beta values to find the one that maximizes the EI of the selected point
    for beta in betas:
        # Acquisition score: EI/cost + beta * sigma/cost
        scores = base_score + beta * expl_component
        
        # Handle non-finite scores for selection purposes
        safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # Find the best candidate according to this beta
        idx_max = np.argmax(safe_scores)
        
        # Calculate Expected Improvement (EI) for this candidate
        # Note: We use the raw EI value (not cost-normalized) to evaluate the "quality"
        # of the choice made by this specific acquisition function.
        mu_x = mu[idx_max]
        sigma_x = sigma_safe[idx_max]
        
        if sigma_x < 1e-9:
            ei_val = max(f_best - mu_x, 0.0)
        else:
            z_x = (f_best - mu_x) / sigma_x
            ei_val = (f_best - mu_x) * norm.cdf(z_x) + sigma_x * norm.pdf(z_x)
            
        # We want to maximize the EI of the selected point
        # This acts as a proxy for "how good is the next point likely to be?"
        # By choosing the beta that leads to the highest EI for the argmax point,
        # we are effectively selecting the most promising exploitation/exploration mix.
        
        if ei_val > best_ei_of_best:
            best_ei_of_best = ei_val
            best_beta = beta
            best_scores = scores
            
    # If for some reason best_scores is None (should not happen), fallback
    if best_scores is None:
        best_scores = base_score + 1.0 * expl_component
        
    # Ensure the returned scores are finite for the final output
    # The task requires 2000 finite scores
    return np.where(np.isfinite(best_scores), best_scores, 0.0)
# EVOLVE-BLOCK-END
