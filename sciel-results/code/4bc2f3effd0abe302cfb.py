"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Structural change: Kriging Believer with adaptive uncertainty scaling,
replacing the linear EI/UCB blend.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _kriging_believer_score(mu, sigma, f_best, cost, beta):
    """
    Kriging Believer acquisition for minimization.
    
    The Kriging Believer method uses the GP's predictive mean as a "believer"
    surrogate and selects points that would be optimal under that surrogate.
    For minimization with a GP, the score is:
        score = -mu + beta * sigma
    
    This is equivalent to LCB (Lower Confidence Bound) for minimization.
    The key structural difference from the previous EI+UCB blend is:
    1. No separate EI calculation (which has scale mismatch issues)
    2. Pure uncertainty-driven exploration via beta*sigma
    3. The beta parameter adapts to the current best value
    
    For Rastrigin's multimodal landscape, this allows the GP uncertainty to
    naturally guide exploration toward unexplored regions where the function
    could potentially be lower.
    """
    # Base score: LCB for minimization
    # We want to maximize this score to find the next point
    # -mu encourages low predicted values
    # +beta*sigma encourages high uncertainty (exploration)
    score = -mu + beta * sigma
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Uses Kriging Believer with adaptive beta based on current best value.
    The cost-aware normalization encourages evaluations that are both
    promising and cheap.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive beta based on f_best
    # Early on (high f_best), we want more exploration (higher beta)
    # Late on (low f_best), we want more exploitation (lower beta)
    # Rastrigin min is 0. Initial f_best is around 5-15.
    # 
    # Use a smooth adaptation:
    # beta = base_beta * (1 + gamma / (1 + f_best))
    # When f_best is large, beta ~ base_beta
    # When f_best approaches 0, beta ~ base_beta * (1 + gamma)
    # 
    # Actually, we want the opposite: more exploration when lost (high f_best),
    # less exploration when close to optimum (low f_best).
    # 
    # Let's use: beta = base_beta * (f_best / (f_best + tau))
    # When f_best is large, beta ~ base_beta
    # When f_best is small, beta ~ base_beta * (f_best / tau) -> small
    # 
    # But we also want a minimum beta to ensure some exploration.
    # 
    # Alternative: beta = min_beta + (max_beta - min_beta) * sigmoid(f_best)
    # 
    # Let's use a simple linear decay with clipping:
    base_beta = 1.5
    min_beta = 0.5
    max_beta = 2.5
    tau = 2.0
    
    # beta decreases as f_best decreases
    # beta = min_beta + (max_beta - min_beta) * (f_best / (f_best + tau))
    # When f_best = 0: beta = min_beta
    # When f_best -> inf: beta -> max_beta
    beta = min_beta + (max_beta - min_beta) * (f_best / (f_best + tau))
    
    # Clip beta to reasonable range
    beta = np.clip(beta, min_beta, max_beta)
    
    # Calculate Kriging Believer score
    kb_score = _kriging_believer_score(mu, sigma, f_best, cost, beta)
    
    # Cost-aware normalization
    # Divide by cost to prefer cheaper evaluations
    scores = kb_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Handle any remaining non-finite values
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
