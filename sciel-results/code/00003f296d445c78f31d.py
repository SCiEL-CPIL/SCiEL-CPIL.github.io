"""Cap set greedy construction with position-weighted zero-bias priority."""
import time
import math

_start_time = None

def _init_global():
    global _start_time
    _start_time = time.time()

def priority(vector, n):
    """
    Priority for greedy cap set construction.
    Higher priority = added earlier.
    
    Heuristic: Weighted sum where positions with digit 0 get higher weights.
    The idea is that 0 is the additive identity, so vectors with zeros in
    "important" positions are less likely to form symmetric 3-term APs.
    
    We assign exponentially decreasing weights to positions, and for each
    position, the contribution depends on the digit:
    - digit 0: +w_i (positive, encourages zeros in high-weight positions)
    - digit 1: 0
    - digit 2: -w_i (negative, discourages 2s in high-weight positions)
    
    This creates a structured preference that breaks symmetry and may
    reduce early AP formation.
    """
    if _start_time is None:
        _init_global()
    
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple deterministic priority
        return float(-sum(vector[i] * (i + 1) for i in range(n)))
    
    # Compute weighted score with exponential position weights
    # Higher positions (later indices) get lower weights
    # digit 0 -> +w, digit 1 -> 0, digit 2 -> -w
    score = 0.0
    for i in range(n):
        w = math.exp(-0.15 * i)  # exponential decay
        d = vector[i]
        if d == 0:
            score += w
        elif d == 2:
            score -= w
        # d == 1: no contribution
    
    # Add a small balance component as tiebreaker
    counts = [0, 0, 0]
    for x in vector:
        counts[x] += 1
    mean = n / 3.0
    variance = sum((c - mean) ** 2 for c in counts)
    score -= 0.01 * variance
    
    return score
