"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Expected Improvement per unit cost.
    
    Strategy:
    - Compute Expected Improvement (EI) for each candidate: E[max(0, f_best - f)]
    - Normalize by cost to get "improvement per unit cost"
    - Add an adaptive exploration bonus proportional to sigma/cost
    - The exploration bonus decays as f_best improves (lower f_best -> less exploration)
    """
    start_time = time.time()
    
    n = len(mu)
    
    # Numerical stability
    sigma_safe = np.clip(sigma, 1e-8, None)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # For minimization: improvement = max(0, f_best - f)
    # E[improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # when f_best > mu, otherwise 0 (approximately)
    
    # z = (f_best - mu) / sigma
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we need to be careful: this formula assumes f_best >= mu
    # For f_best < mu, EI is still positive but smaller
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative; clip to 0
    ei = np.maximum(ei, 0.0)
    
    # EI per unit cost
    ei_cost = ei / cost_safe
    
    # Adaptive exploration bonus
    # Decay exploration as f_best improves
    # Use log(1 + max(0, -f_best)) for smoother decay
    # When f_best is large (early), explore more; when f_best is small (late), exploit more
    # f_best is the minimum observed value, so it starts high and decreases
    # We want beta to decrease as f_best decreases
    # beta = 2.0 / (1.0 + log(1.0 + abs(f_best)))
    # But f_best can be negative, so use abs
    beta = 2.0 / (1.0 + np.log1p(abs(f_best)))
    
    # Ensure beta has a floor to maintain some exploration
    beta = max(beta, 0.1)
    
    # Exploration bonus: beta * sigma / cost
    exploration_bonus = beta * sigma_safe / cost_safe
    
    # Combine
    scores = ei_cost + exploration_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Fallback: simple -mu/cost
        simple_scores = -mu / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
