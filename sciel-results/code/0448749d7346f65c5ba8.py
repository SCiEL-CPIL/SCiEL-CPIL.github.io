"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and has a minimum floor to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term
    beta = 0.5
    ucb = mu + beta * sigma
    
    # Hybrid: blend EI and UCB based on uncertainty
    # When sigma is large, weight UCB more; when sigma is small, weight EI more
    # Normalize sigma to get a weight in [0, 1]
    max_sigma = np.max(sigma)
    if max_sigma > 0:
        sigma_weight = sigma / max_sigma
    else:
        sigma_weight = np.zeros_like(sigma)
    
    # Blend factor: more exploration when uncertain
    blend = 0.5 + 0.5 * sigma_weight
    
    # Combine EI and UCB scores
    # Scale UCB to be comparable to EI (both are "improvement" metrics)
    # EI is in objective units (lower is better), so we want high EI = good
    # UCB gives upper confidence bound; for minimization, we want low f, 
    # so we use -UCB as a score where higher is better? No.
    # Actually, for minimization:
    # EI = E[max(f_best - f(x), 0)] > 0 when improvement expected
    # UCB = mu + beta*sigma is an upper bound on f(x). Lower is better.
    # So for UCB-based score, we want to minimize UCB, meaning score = -UCB
    # But EI is already a "benefit" score (higher is better)
    
    # Let's use a different approach: 
    # score = w_ei * EI + w_ucb * (f_best - UCB) where (f_best - UCB) is optimistic improvement
    # But (f_best - UCB) can be negative. Let's use EI-like formulation.
    
    # Alternative: Use EI with adaptive beta
    # EI_beta = (f_best - mu) * Phi(z) + sigma * phi(z) is standard EI
    # We can modify the exploration by scaling sigma in the z-score
    
    # Simpler approach: Use cost-aware EI with enhanced exploration
    # The key insight from inspiration: hybrid UCB+EI with adaptive weighting
    
    # Let's compute both and blend
    # EI is already computed
    # For UCB contribution to "improvement": optimistic improvement = f_best - (mu - beta*sigma)
    # This is the best-case improvement if we assume f(x) could be as low as mu - beta*sigma
    optimistic_improvement = f_best - (mu - beta * sigma)
    optimistic_improvement = np.maximum(optimistic_improvement, 0.0)
    
    # Blend EI and optimistic improvement
    score_benefit = (1 - blend) * ei + blend * optimistic_improvement
    
    # Cost-aware penalty: use 1/cost to strongly prefer cheaper points
    cost_penalty = 1.0 / cost
    
    # Final score
    score = score_benefit * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
