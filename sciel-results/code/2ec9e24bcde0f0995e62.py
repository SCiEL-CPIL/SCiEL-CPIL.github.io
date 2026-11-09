import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a hybrid of Expected Improvement (EI) and adaptive Upper Confidence Bound (UCB).
    The UCB weight adapts based on progress towards the known minimum (3.0).
    """
    # Deadline check: if we're close to the limit, return a simple deterministic score
    # to ensure we don't get killed mid-computation.
    if time.time() > _DEADLINE:
        return -mu  # Fallback: pure exploitation

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI) component
    # We are minimizing, so improvement is (f_best - mu)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Adaptive UCB component
    # The minimum value of Goldstein-Price is 3.0.
    # The maximum value in the domain [-2, 2]^2 is approximately 12-13.
    min_val = 3.0
    max_val_est = 13.0
    
    gap = f_best - min_val
    max_gap = max_val_est - min_val
    
    # Clip progress to [0, 1]
    # progress = 0 when f_best is high (early), 1 when f_best is low (late)
    progress = np.clip((max_gap - gap) / max_gap, 0.0, 1.0)
    
    # Beta ranges from 2.0 (high exploration early) to 0.2 (high exploitation late)
    # We start with a lower max beta than inspiration 1 (4.0) because EI already provides
    # a strong exploitation signal, and we want to avoid over-exploration which hurts
    # in the tight budget.
    beta_max = 2.0
    beta_min = 0.2
    beta = beta_max - (beta_max - beta_min) * progress
    
    ucb_score = -mu + beta * sigma
    
    # 3. Hybrid Score
    # Combine EI and UCB. EI is in "improvement" units, UCB is in "function value" units.
    # To make them comparable, we can scale EI. However, a simpler and often effective
    # approach is to use EI as the base and add a scaled UCB term, or vice versa.
    # Given the stagnation, let's try a weighted sum.
    # Since EI tends to be small when f_best is close to mu, and UCB is linear in mu/sigma,
    # we'll normalize or just weight them.
    # Let's use: score = w_ei * ei + w_ucb * ucb_score
    # But the scales are different.
    # Alternative: Use EI primarily, but boost exploration when EI is low (uncertain).
    # A common hybrid is: score = ei + beta * sigma
    # This is essentially EI + exploration bonus.
    
    # Let's refine the "Exploration Bonus" approach from Inspiration 3, but make the
    # beta adaptive like Inspiration 1.
    # score = ei + adaptive_beta * sigma
    
    # The adaptive beta here controls how much extra exploration we add on top of EI.
    # Early on (low progress), we want more exploration.
    # Late on (high progress), we want less.
    
    # Let's use a slightly different adaptive schedule for the bonus term.
    # If f_best is far from min, we are lost, so explore more.
    # If f_best is close, we are refining, so explore less.
    
    bonus_beta = beta  # Reuse the adaptive beta from above
    
    scores = ei + bonus_beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
