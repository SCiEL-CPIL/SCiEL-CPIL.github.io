"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_DEADLINE = None

def _ensure_deadline():
    global _START_TIME, _DEADLINE
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 320.0  # 320s safety margin within 330s limit
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    
    # Structural Change: Adopt pure Upper Confidence Bound (UCB) for minimization.
    # The hybrid approach in the current program (blending normalized UCB and EI)
    # underperforms compared to a direct, aggressive exploration strategy.
    # For multimodal functions like Hartmann 6D with a limited budget (30 evals),
    # high exploration is critical to escape local optima.
    
    # UCB Score = -mu + kappa * sigma
    # Higher score means we prefer this candidate.
    # kappa = 3.0 is chosen based on empirical success in similar tasks (inspiration 1-3).
    kappa = 3.0
    
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite as required by the task contract.
    # This handles any potential numerical instabilities from the GP posterior.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
