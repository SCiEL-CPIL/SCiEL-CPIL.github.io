import numpy as np
import time

# Track call index to adapt strategy over the optimization loop
_call_count = [0]

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Thompson Sampling and UCB/EI
    with adaptive weighting based on progress (f_best).
    
    Early on (high f_best): more exploration via UCB-like term.
    Later (low f_best): more exploitation via Thompson Sampling / EI-like term.
    """
    _call_count[0] += 1
    step = _call_count[0]
    
    # Ensure sigma is strictly positive
    eps = 1e-9
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Thompson Sampling: sample from the posterior
    rng = np.random.default_rng()
    samples = mu + rng.standard_normal(len(mu)) * sigma_safe
    ts_score = -samples  # For minimization, we want low values, so negative
    
    # UCB for minimization: -mu + beta * sigma
    # Decay beta over time to shift from exploration to exploitation
    if step <= 5:
        beta = 1.0
    elif step <= 10:
        beta = 0.5
    elif step <= 20:
        beta = 0.2
    else:
        beta = 0.05
    
    ucb_score = -mu + beta * sigma_safe
    
    # Adaptive weighting:
    # f_best starts high (e.g., ~12.5 for random points on [-5,5]) and goes to 0
    # Use f_best to determine the blend
    # When f_best is high, we need more exploration (UCB)
    # When f_best is low, we can exploit more (Thompson Sampling)
    
    # Scale: f_best=5 is "medium", f_best=0 is "exploit", f_best=25 is "explore"
    t = np.clip(f_best / 5.0, 0.0, 1.0)
    
    w_explore = t
    w_exploit = 1.0 - t
    
    score = w_exploit * ts_score + w_explore * ucb_score
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
