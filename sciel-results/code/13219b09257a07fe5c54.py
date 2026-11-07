"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Bayesian optimization.
    
    We draw a single sample from the GP posterior: y = mu - sigma * z,
    where z ~ N(0, 1). Then we score each candidate by y / cost,
    preferring low values (minimization) and low cost.
    
    To make this deterministic per call (so the evaluator can pick the best),
    we use a fixed random seed derived from the current state to ensure
    reproducibility within a single call, but the key insight is that
    Thompson Sampling naturally balances exploration and exploitation.
    
    However, since we need to return 2000 scores and the evaluator picks
    the argmax, we can use a deterministic approach that mimics Thompson
    Sampling's behavior: we can use a single fixed z-sample for all points,
    or better, we can use a hybrid approach.
    
    Actually, let's use a proper Thompson Sampling approach:
    - Draw one z ~ N(0,1) per candidate (or use a deterministic proxy)
    - Score = (mu - sigma * z) / cost
    
    But since we need to return scores for 2000 candidates and the evaluator
    picks the best, we should make the scores deterministic. Let's use:
    score = (mu - beta * sigma) / cost
    
    where beta is a fixed exploration parameter. This is like a single
    Thompson Sampling draw with a fixed z = beta.
    
    Alternatively, we can use a time-varying or state-varying beta to
    balance exploration/exploitation.
    
    Let's use: score = (mu - sigma * z) / cost where z is drawn from N(0,1)
    with a fixed seed based on the current cost and f_best to ensure
    reproducibility.
    
    Actually, the cleanest approach: use a fixed z value that adapts to
    the optimization state. Early on, use larger z (more exploration);
    later, use smaller z (more exploitation).
    
    Let's compute a progress indicator and adjust z accordingly.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Determine exploration level based on total cost spent
    total_cost = np.sum(cost)
    
    # As total cost increases, reduce exploration
    # Start with high exploration (z ~ 1.0) and decay to low (z ~ 0.1)
    # Use a smooth decay
    exploration = 1.0 / (1.0 + total_cost / 10.0)
    
    # Draw a single z-sample for all candidates (same z for all)
    # This makes the score deterministic given the state
    # Use a fixed seed based on state for reproducibility
    rng = np.random.default_rng(seed=int(total_cost * 1000) % (2**32))
    z = rng.standard_normal()
    
    # Clip z to avoid extreme values
    z = np.clip(z, -3.0, 3.0)
    
    # Adjust z based on exploration level
    z_adjusted = z * exploration + 0.1 * (1 - exploration)
    
    # Thompson Sampling score: (mu - sigma * z) / cost
    # Lower is better, so we negate for "higher = better" scoring
    scores = (mu - sigma * z_adjusted) / cost
    
    # Negate so that higher score = better (lower objective / cost)
    scores = -scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
