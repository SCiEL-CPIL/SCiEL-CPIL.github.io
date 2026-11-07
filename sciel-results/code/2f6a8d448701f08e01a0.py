"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills at 330s. Keep a safety margin.
_DEADLINE = time.time() + 300  # 300s budget with 30s margin

def _check_deadline():
    """Return True if we should stop working (safety margin exceeded)."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) strategy for global optimization.
    # Unlike Expected Improvement (EI) which can be overly exploitative,
    # LCB explicitly balances exploitation (low mu) and exploration (high sigma).
    # This is critical for multimodal landscapes like Hartmann 6D with a small budget.
    
    # We want to maximize the returned score.
    # The LCB value is mu - kappa * sigma (to be minimized).
    # So we return the negative of that: -mu + kappa * sigma.
    
    # Adaptive kappa: start with aggressive exploration, decay toward exploitation
    # as uncertainty decreases relative to the function scale.
    # This smooths the transition between exploration and exploitation.
    
    # Normalize sigma to get a relative uncertainty measure
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    
    # If sigma is very small everywhere, we're in exploitation mode
    # If sigma is large, we're in exploration mode
    if sigma_max > 1e-10:
        # Relative uncertainty: how much uncertainty remains
        rel_uncertainty = sigma_max / (abs(f_best) + 1e-10)
        # Clamp to [0, 1] for stable behavior
        rel_uncertainty = np.clip(rel_uncertainty, 0.0, 1.0)
        
        # Kappa ranges from ~1.0 (exploit) to ~3.5 (explore)
        # Higher when uncertainty is high relative to best value
        kappa = 1.0 + 2.5 * rel_uncertainty
    else:
        kappa = 1.0  # Pure exploitation when no uncertainty
    
    # Safety: ensure kappa stays in a reasonable range
    kappa = np.clip(kappa, 0.5, 4.0)
    
    scores = -mu + kappa * sigma
    
    # Safety: ensure all scores are finite.
    # If the GP produces non-finite values (unlikely but possible), 
    # we assign a very low score so they are not selected.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
