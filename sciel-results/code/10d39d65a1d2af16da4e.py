"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: although this function is vectorized and fast,
    # we check for safety as per instructions.
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score (e.g., just -mu)
        # to force a selection quickly.
        return -mu
    
    # Adaptive beta: start with high exploration, decay to exploitation.
    # Michalewicz 2D is multimodal with 2 global minima.
    # Early on, we need to explore broadly; later, exploit near best.
    # beta starts at 4.0 and decays to 1.0
    beta = 1.0 + 3.0 * np.exp(-_eval_count / 5.0)
    
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the acquisition score.
    # Low mu (good mean) and High sigma (good uncertainty) should yield high score.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Add small stochastic noise to prevent getting stuck in local optima
    # when acquisition scores are similar. This encourages diversity in selection.
    noise_scale = 0.01 * np.std(mu) + 1e-6
    rng = np.random.default_rng(_eval_count)
    noise = rng.normal(0, noise_scale, size=mu.shape)
    scores = scores + noise
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
