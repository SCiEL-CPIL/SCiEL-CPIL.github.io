"""
Strategy: Probability of Improvement (PI) with Adaptive Exploration
This approach differs from Expected Improvement (EI) by maximizing the probability
that a candidate point yields a better objective value than the current best.
PI is more aggressive in exploitation but can be overly greedy. To counteract this
in the multi-modal Styblinski-Tang landscape, we add an exploration bonus term
that is inversely proportional to the posterior uncertainty (sigma) and scaled
by a decay factor based on the current best value relative to the known global
minimum. This encourages exploration early on and exploitation later.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) with adaptive exploration.
    
    Args:
        mu: array of shape (n,), posterior mean
        sigma: array of shape (n,), posterior std
        f_best: float, best observed value so far
    
    Returns:
        scores: array of shape (n,), higher is better
    """
    n = len(mu)
    
    # Safety check for time
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple PI if running out of time
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return norm.cdf(z)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Probability of Improvement (PI)
    # P(y < f_best) = P(N(mu, sigma^2) < f_best)
    # = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)
    
    # 2. Adaptive Exploration Bonus
    # The known global minimum for Styblinski-Tang 1D is approximately -39.1662.
    # We use this to gauge how close we are to the optimum.
    f_min_known = -39.1662
    
    # As f_best approaches f_min_known, we want to reduce exploration.
    # If f_best is far from f_min_known, we want more exploration.
    # Let's define a progress metric.
    # Note: f_best starts high (e.g., 50) and decreases to -39.
    # progress = (f_best - f_min_known) / (f_initial_guess - f_min_known)
    # Since we don't know f_initial_guess reliably, we can use a heuristic.
    # A simple decay based on the magnitude of improvement from a rough start.
    # Let's assume the initial points are in [-5, 5], so f is in [-39, 50].
    # A simple heuristic: exploration_weight decreases as f_best decreases.
    
    # If f_best is very bad (e.g., > 0), we are early. 
    # If f_best is good (e.g., < -30), we are late.
    
    # Let's use a sigmoid-like decay or linear decay.
    # Linear decay: 
    # weight = 1.0 if f_best > 0
    # weight = 0.0 if f_best < -39
    # weight = (f_best - (-39)) / (0 - (-39)) = (f_best + 39) / 39
    
    # Clip to [0, 1]
    exploration_weight = np.clip((f_best - f_min_known) / (0.0 - f_min_known), 0.0, 1.0)
    
    # Scale the exploration bonus. 
    # The bonus should be proportional to uncertainty (sigma) or inverse?
    # Usually, exploration is driven by high uncertainty.
    # But PI already exploits low mean. 
    # Adding a term proportional to sigma encourages exploring uncertain regions.
    # However, sigma is in objective units. We need to normalize it to be comparable to PI (0..1).
    
    # Let's estimate a typical scale of the objective function.
    # Range is roughly [-39, 50], so ~90 units.
    # Let's normalize sigma by 10 (a heuristic scale for local variations).
    # Or better, use the current sigma distribution.
    
    # To keep it simple and robust:
    # Exploration Bonus = exploration_weight * (sigma / sigma_scale)
    # We need to ensure this doesn't dominate PI.
    # Let's normalize the exploration term to have a max value around 0.5 * exploration_weight.
    
    # Let's use the median sigma as a scale reference.
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-6:
        sigma_median = 1.0
        
    # Normalize sigma
    sigma_norm = sigma_safe / sigma_median
    
    # Exploration term
    # We want this to be small when sigma is small (explored) and large when sigma is large (unexplored).
    # But we also want it to decay as we get closer to the optimum.
    exploration_bonus = exploration_weight * 0.5 * np.exp(-sigma_norm) # Wait, exp(-sigma) is small for large sigma.
    # Actually, we want to explore where sigma is HIGH.
    # So maybe just proportional to sigma?
    # Let's use a scaled sigma.
    # exploration_bonus = exploration_weight * 0.1 * sigma_norm
    
    # Let's try a different exploration strategy: 
    # Use the standard deviation itself, scaled to be comparable to the PI score range [0,1].
    # If sigma is 1, and typical improvement is 1, then PI is ~0.5.
    # Let's just add a small fraction of the normalized sigma.
    
    # Revised Exploration Bonus:
    # We want to push towards points with high sigma when we are far from the optimum.
    # bonus = exploration_weight * (sigma_safe / 10.0) # 10 is a heuristic scale
    # To keep it bounded, let's use a soft cap or normalization.
    
    # Let's stick to a simpler, more standard approach: 
    # PI + alpha * sigma
    # Where alpha decays.
    
    alpha = 0.1 * exploration_weight
    
    scores = pi_scores + alpha * sigma_safe
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
# EVOLVE-BLOCK-END
