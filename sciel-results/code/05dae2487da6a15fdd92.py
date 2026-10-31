"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Polarization Score
    - Encourages remaining capacities to be close to 0 (full) or close to the 
      maximum remaining capacity in the system (fresh).
    - This dynamically adapts between Best-Fit (early game/high M) and 
      Worst-Fit (late game/low M) without a static item-size threshold.
    - Score = -min(g, M - g), where g is the gap after placement and M is 
      the max remaining capacity in the system.
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    # M represents the "freshness" level of the system
    # Using the max remaining capacity of all bins (including infeasible ones)
    # provides a global context for what "empty" means in the current state.
    M = np.max(bins)
    
    # Calculate gaps for feasible bins
    g = remaining_after[feasible]
    
    # Polarization score: prefer g close to 0 or g close to M
    # min(g, M - g) is small when g is near 0 or M
    # We want high scores, so we negate it
    # Note: M - g is always >= 0 because g <= bins[i] <= M
    scores_feasible = -np.minimum(g, M - g)
    
    scores = np.full(len(bins), -np.inf)
    scores[feasible] = scores_feasible
    
    # Add a tiny deterministic tie-breaker favoring lower indices
    indices = np.arange(len(bins))
    scores = scores - 1e-10 * indices
    
    return scores
# EVOLVE-BLOCK-END
