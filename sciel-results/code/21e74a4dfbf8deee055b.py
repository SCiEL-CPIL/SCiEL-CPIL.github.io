"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds before the 330s limit to stop work

def acquisition(mu, sigma, f_best):
    # Check time budget
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Fallback to simple EI if we are running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    n = len(mu)
    
    # 1. Compute base UCB scores (kappa=1.5 was measured to help)
    kappa = 1.5
    ucb = mu - kappa * sigma
    
    # 2. Compute EI scores for diversity penalty
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 3. Apply diversity penalty
    # For each candidate i, find the max EI among all j != i.
    # This penalizes candidates that are "redundant" with high-EI neighbors.
    # Since we don't have coordinates, we approximate spatial redundancy by
    # assuming candidates with high EI are "important" and we want to diversify 
    # away from the single best EI point? 
    # Actually, without coordinates, we can't compute true distance.
    # However, the task says "mu and sigma have shape (2000,)". 
    # The candidates are drawn uniformly. 
    # A structural change in *representation* without coordinates is difficult.
    # But wait, the prompt says "No coordinates are passed."
    # So I cannot compute distances between the 2000 candidates.
    
    # Re-evaluating the "Diversity" idea without coordinates.
    # If I can't compute distances, I can't do true diversity penalty on the set.
    
    # Alternative Structural Change: 
    # The current approach is deterministic ranking.
    # A different design family: **Thompson Sampling**.
    # Instead of maximizing a deterministic score (EI/UCB), we sample from the posterior.
    # But we only have mu and sigma, not the full GP.
    # We can approximate Thompson Sampling by sampling from N(mu, sigma) for each point?
    # No, that's just sampling the objective.
    
    # Let's look at the "Inspiration 2" which scored 1.1513. It mentioned "adaptive UCB".
    # The current program is static EI.
    # LCB with kappa=1.5 scored 1.044.
    
    # What is a MATERIALLY DIFFERENT strategy?
    # 1. EI (Current)
    # 2. LCB (Tried, slightly better)
    # 3. PI (Probability of Improvement) - usually worse than EI.
    # 4. Max-Value Entropy Search (MVES) - requires coordinates.
    # 5. **Kriging Believer / Deterministic UCB with adaptive kappa**.
    
    # Let's try **Adaptive UCB** based on the number of evaluations.
    # We don't know the number of evaluations directly, but we can infer it from 
    # the spread of f_best or just use a heuristic.
    # Actually, we can track the call count in the module state.
    
    global _call_count
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    
    acquisition._call_count += 1
    step = acquisition._call_count
    
    # Adaptive kappa: Start high for exploration, decrease for exploitation.
    # Total evaluations = 30.
    # Initial 5 are random.
    # Steps 1..25 (corresponding to evals 6..30)
    # Let's use a schedule: kappa = 2.0 / (1 + 0.1 * step)
    # Or simpler: linear decay from 2.0 to 0.5.
    
    if step <= 5:
        kappa = 2.0
    elif step <= 10:
        kappa = 1.5
    elif step <= 20:
        kappa = 1.0
    else:
        kappa = 0.5
        
    scores = mu - kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
