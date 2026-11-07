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
    beta = 1.0 + 3.0 * np.exp(-_eval_count / 4.0)
    
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the acquisition score.
    # Low mu (good mean) and High sigma (good uncertainty) should yield high score.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Add small stochastic perturbation to break ties and prevent cycling.
    # The perturbation is scaled by sigma so it's more exploratory in uncertain regions.
    # This helps escape local optima in the multimodal Michalewicz landscape.
    rng = np.random.default_rng()
    noise_scale = 0.1 * sigma
    scores = scores + rng.normal(0, 1, size=scores.shape) * noise_scale
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
