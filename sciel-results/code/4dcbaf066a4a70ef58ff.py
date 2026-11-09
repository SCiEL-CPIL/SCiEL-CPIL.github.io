"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

# Explicit deadline guard: evaluator kills at 330s wall-clock.
_T0 = time.time()
_DEADLINE = _T0 + 285.0  # 45s safety margin

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    
    Strategy (refined within the balance-based family):
    1. Quadratic edge distance: Penalize deviation from n/3 per symbol quadratically.
       This better separates safe edge vectors from risky midpoints.
    2. Symbol diversity bonus: Reward vectors that use all three symbols.
       This reduces local clustering in the cube.
    3. Position-dependent hash tie-break: Structured spreading across the cube.
    """
    # Deadline guard: if we're running long, return a safe constant
    if time.time() > _DEADLINE:
        return 0.0
    
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Target is n/3
    target = n / 3.0
    
    # Quadratic deviation from balance (penalizes midpoints more strongly)
    dev0 = c0 - target
    dev1 = c1 - target
    dev2 = c2 - target
    quadratic_dev = dev0 * dev0 + dev1 * dev1 + dev2 * dev2
    
    # Symbol diversity bonus: reward using all three symbols
    num_symbols = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    diversity_bonus = num_symbols * 0.1
    
    # Base score: higher is better
    score = -quadratic_dev + diversity_bonus
    
    # Position-dependent hash tie-break for structured spreading
    # Use a weighted sum with position-dependent weights to break ties
    tie_break = 0.0
    for i, v in enumerate(vector):
        # Alternate weights to create more structured tie-breaking
        weight = (i + 1) if (i % 2 == 0) else -(i + 1)
        tie_break += weight * v
    score += tie_break * 1e-10
    
    return score

# EVOLVE-BLOCK-END
