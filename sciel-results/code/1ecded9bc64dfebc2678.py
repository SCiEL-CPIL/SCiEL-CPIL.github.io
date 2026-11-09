"""Cap set priority function: structured greedy ordering for large cap sets."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def priority(vector, n):
    """
    Priority for greedy cap set construction.
    
    Strategy: 
    - Primary: minimize number of 2s (vectors in {0,1}^n form a cap set of size 2^n)
    - Secondary: structured lexicographic ordering to maximize successful insertions
      of vectors with 1 or 2 twos
    
    This gives the greedy algorithm a strong base (all {0,1} vectors) and then
    attempts to extend with carefully ordered vectors containing 2s.
    """
    # Count the number of 2s in the vector
    num_twos = sum(1 for v in vector if v == 2)
    
    # Primary priority: fewer 2s is better (higher priority)
    # Scale by a large factor so it dominates
    base_priority = -num_twos * 10000.0
    
    # Secondary: lexicographic value (treat vector as base-3 number)
    # Higher lexicographic value gets higher priority within same num_twos
    lex_value = 0.0
    for i, v in enumerate(vector):
        lex_value += v * (3 ** (n - 1 - i))
    
    # Tertiary: prefer vectors where 2s are in "spread out" positions
    # (less likely to create 3-APs with other vectors)
    # Use the positions of 2s as a tiebreaker
    two_positions = [i for i, v in enumerate(vector) if v == 2]
    spread_bonus = 0.0
    if len(two_positions) >= 2:
        # Prefer 2s that are far apart (larger minimum distance)
        min_dist = min(abs(two_positions[j+1] - two_positions[j]) for j in range(len(two_positions)-1))
        spread_bonus = min_dist * 0.1
    
    return base_priority + lex_value + spread_bonus
# EVOLVE-BLOCK-END
