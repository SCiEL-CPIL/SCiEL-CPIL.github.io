"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # 300s limit, keep margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling on Value per Unit Cost (VPUC).
    
    Instead of a deterministic score like EI/cost or PoI/cost, we sample from the 
    posterior distribution of the objective value for each candidate and compute 
    the resulting VPUC. We then select the candidate that maximizes the sampled VPUC.
    
    This approach naturally balances exploration (high sigma leads to diverse samples) 
    and exploitation (high probability of low f leads to high VPUC) while strictly 
    accounting for the cost of evaluation.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        # Fallback: return simple EI/cost if running out of time
        # This is a safety net, though the main logic is fast
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        poi = norm.cdf(z)
        score = poi / cost
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Cost is given as shape (2000,)
    # Cost formula: 1 + mean(x_scaled). x is in [0,10]^4, scaled to [0,1] means x/10.
    # The input 'cost' is the pre-computed cost for each candidate.
    
    # We want to maximize E[ (f_best - f(x)) / cost(x) ] subject to f(x) < f_best?
    # Actually, standard cost-aware BO often maximizes the expected improvement per unit cost.
    # Let's use a Thompson Sampling approach on the "Value" metric.
    
    # Define the utility we want to maximize: 
    # If we evaluate x, we get value f(x). The "gain" is max(0, f_best - f(x)).
    # The "cost" is cost(x).
    # We want to maximize Gain / Cost.
    
    # Thompson Sampling:
    # 1. Sample f_sample ~ N(mu, sigma) for all candidates.
    # 2. Compute gain_sample = max(0, f_best - f_sample)
    # 3. Compute vpuc_sample = gain_sample / cost
    # 4. Select the candidate with the highest vpuc_sample.
    
    # However, the function signature requires returning a score for ALL 2000 candidates 
    # such that the evaluator picks the best one. 
    # Usually, TS is used by sampling once and picking the argmax. 
    # To fit the "return scores for all" interface, we can return the sampled VPUC values 
    # from a single draw. The evaluator will pick the argmax of these returned scores.
    # This is equivalent to standard TS selection.
    
    # Generate samples from the posterior
    # mu and sigma are in objective units.
    # We need to be careful with sigma=0.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw one sample per candidate
    samples = mu + sigma_safe * np.random.normal(size=mu.shape)
    
    # Calculate the improvement (gain)
    # We only care about improvements. If sample > f_best, gain is 0.
    gains = np.maximum(0, f_best - samples)
    
    # Calculate Value Per Unit Cost (VPUC)
    # cost is positive (1 + mean(x_scaled) >= 1)
    vpuc = gains / cost
    
    # The evaluator will pick the candidate with the highest vpuc.
    # To make this robust, we return the vpuc as the score.
    
    # Handle any numerical issues
    vpuc = np.nan_to_num(vpuc, nan=0.0, posinf=0.0, neginf=0.0)
    
    return vpuc
# EVOLVE-BLOCK-END
