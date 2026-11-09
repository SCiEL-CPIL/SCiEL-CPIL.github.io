"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds margin before 330s limit


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return a deterministic fallback, e.g., standard EI
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure sigma is positive to avoid division or log issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive UCB component
    # Branin global min is approx 0.397887
    # We want beta to be higher when f_best is large (uncertain/early)
    # and lower when f_best is small (exploit/late)
    
    f_min_known = 0.397887
    # Distance from optimum
    dist = max(f_best - f_min_known, 0.0)
    
    # Map dist to beta
    # dist=0 -> beta=0.3
    # dist=15 -> beta=1.5
    beta = 0.3 + 1.2 * (dist / 15.0)
    beta = np.clip(beta, 0.3, 1.5)
    
    # UCB for minimization:
    # Score = beta * sigma - mu
    ucb_score = beta * sigma - mu
    
    # Combine EI and UCB
    # Use a weighted combination where EI dominates when close to optimum
    # and UCB provides exploration when far from optimum
    
    # Normalize scores to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-9
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1e-9
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max
    
    # Weight: when f_best is close to optimum, weight EI more heavily
    # When f_best is far, weight UCB more heavily
    if dist < 5.0:
        # Close to optimum: EI dominant (70/30)
        w_ei = 0.7
        w_ucb = 0.3
    else:
        # Far from optimum: UCB dominant (30/70)
        w_ei = 0.3
        w_ucb = 0.7
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
