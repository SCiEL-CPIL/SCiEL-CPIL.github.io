"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_START_TIME = None

def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _DEADLINE is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    _ensure_deadline()
    
    # Best-fit: prefer bins that leave the least remaining capacity.
    # Score is negative of remaining space, so higher score = less remaining space
    remaining = bins - item
    return -remaining
# EVOLVE-BLOCK-END
