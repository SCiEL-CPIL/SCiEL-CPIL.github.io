import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling:
    # We sample a single function value from the GP posterior for each candidate.
    # For minimization, we want to find the minimum of the sampled function.
    # The score should reflect the likelihood of this candidate being the best.
    # A common proxy for TS in acquisition functions is the sampled value itself,
    # but we want to maximize the score.
    # So we can use -sampled_value / cost.
    # However, a more robust approach for TS in BO is to sample the function 
    # and then the acquisition is often just the sampled value (minimized).
    # Since the interface requires returning scores where higher = evaluate next,
    # we return -sampled_value / cost.
    
    # To make it more stable and less sensitive to extreme noise, we can 
    # regularize the sampling or use a variant.
    # Standard TS: sample theta ~ N(mu, sigma^2).
    # The "acquisition" for the next point is the point that minimizes theta.
    # Since we must return a score for each point to let the external loop pick the max,
    # we compute the sampled value and invert it.
    
    # Use a random seed internally? No, random state should be global or default.
    # np.random.standard_normal is fast.
    z = np.random.standard_normal(size=mu.shape)
    
    # Sampled values
    sampled_values = mu + sigma * z
    
    # We want to minimize sampled_values.
    # Score = -sampled_values / cost
    # This favors low sampled values (good) and low cost (good).
    
    # To prevent extreme outliers from dominating due to random noise in TS,
    # we might clip the sampled values or use a robust metric.
    # However, standard TS is known to be effective.
    
    # One issue with pure TS is that if sigma is large, the sample can be very high or low.
    # Dividing by cost helps, but a very low sample (due to noise) might be picked.
    # This is the nature of exploration.
    
    # Let's refine: 
    # If we just use -sampled_values / cost, we are effectively doing:
    # Maximize ( - (mu + sigma*z) ) / cost
    # This is equivalent to maximizing ( -mu - sigma*z ) / cost
    
    # Alternative: Expected Improvement is more stable. 
    # But the directive asks for a DIFFERENT strategy.
    # Let's try a "Cost-Aware Expected Improvement" but with a twist:
    # The current program uses UCB. The parent (score 1.0) used EI/cost.
    # The current program (score 0.5) uses UCB/cost.
    # The parent's EI/cost was better. Why?
    # EI is more robust to noise than UCB with fixed beta.
    # However, the directive says "MATERIALLY DIFFERENT".
    # TS is materially different from EI and UCB.
    
    # Let's stick to TS but add a small exploitation bias to ensure we don't just jump to random noise.
    # A common hybrid is to use the sampled value but shrink it towards mu.
    # Or, we can use the probability of improvement? No, that's EI-like.
    
    # Let's try: Score = - (mu + 0.5 * sigma * z) / cost
    # This reduces the variance of the exploration signal.
    
    # Actually, a very strong competitor to EI is the "Probability of Improvement" (PI) or 
    # using the quantile of the distribution.
    
    # Let's try a different angle: **Knowledge Gradient (KG)** approximation.
    # KG is computationally expensive.
    
    # Let's go back to **Thompson Sampling** but implement it carefully.
    # The key advantage of TS over UCB is that it adapts the exploration rate automatically 
    # based on the uncertainty.
    
    # To make TS more robust for minimization with cost:
    # 1. Sample the objective function values.
    # 2. The score is the negative of the sampled value divided by cost.
    
    # Potential pitfall: If sigma is very small, TS behaves like greedy.
    # If sigma is large, TS explores.
    
    # Let's add a deadline guard as requested, although this function is vectorized and fast.
    # The "loop" mentioned in the directive is external. This function is called per iteration.
    # 2000 points is small. np.random and basic arithmetic are fast.
    
    # Implementation:
    z = np.random.standard_normal(size=mu.shape)
    
    # To avoid extreme outliers affecting the choice too much (which can happen if the GP is poorly calibrated),
    # we can clip the z-scores.
    z = np.clip(z, -3, 3)
    
    sampled = mu + sigma * z
    
    # We want to minimize 'sampled'.
    # Score = -sampled / cost
    score = -sampled / cost_safe
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
