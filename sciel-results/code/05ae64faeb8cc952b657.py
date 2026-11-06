"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    
    Strategy:
    1. Identify that the subset {0, 2}^n forms a cap set (no 3-term APs).
       Proof: If a, c in {0, 2}^n, then b = (a+c)/2 has coordinates in {0, 1, 2}.
       If a_i != c_i, then b_i = 1. If a_i == c_i, then b_i = a_i.
       For b to be in {0, 2}^n, we must have b_i != 1 for all i, which implies a_i == c_i for all i.
       Thus a = c. Since we require distinct a, b, c, no AP exists in {0, 2}^n.
    2. Any vector containing a '1' will form an AP with pairs of vectors in {0, 2}^n.
       Specifically, if b has a '1' at position i, and a, c differ at i (0 and 2) and match elsewhere,
       they form an AP. Since the greedy algorithm fills {0, 2}^n first, all such a, c will be present,
       causing b to be rejected.
    3. Therefore, the optimal static ordering is to prioritize all vectors in {0, 2}^n.
       This guarantees a cap set of size 2^n.
       Adding any vector with a '1' would require removing multiple vectors from {0, 2}^n,
       which is a net loss (1 gained vs 2^k lost for k ones).
    
    Implementation:
    - Vectors with no '1's get high priority.
    - Vectors with '1's get low priority.
    - Deterministic tiebreaker for consistent ordering.
    """
    # Check for presence of 1
    has_one = 1 in vector
    
    if not has_one:
        # High priority: These vectors form a valid cap set among themselves.
        score = 1000.0
    else:
        # Low priority: These will likely be rejected if the {0,2}^n set is full.
        # If for some reason a {0,2} vector is missing, this might be added,
        # but it's a suboptimal trade-off.
        score = -1000.0
        
    # Deterministic tiebreaker
    # Use a weighted sum to ensure unique ordering for identical types
    tiebreak = 0
    for i, v in enumerate(vector):
        tiebreak += v * (i + 1)
    
    # Small epsilon to not override the main score
    score += tiebreak * 0.0001
    
    return score
# EVOLVE-BLOCK-END


def _main():
    """
    Main entry point called by the evaluator.
    """
    pass

# The actual entry point that the evaluator calls
# Based on the task description, the evaluator calls priority() and does
# the greedy selection itself. So I just need to provide a good priority.

def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    Higher priority = added earlier.
    
    Strategy: Favor vectors in {0, 2}^n as they form a maximal cap set of size 2^n.
    Vectors containing 1s are less desirable as they conflict with multiple {0, 2}^n vectors.
    """
    # Check if vector contains any 1s
    # If no 1s, it's in {0, 2}^n
    has_one = False
    for v in vector:
        if v == 1:
            has_one = True
            break
            
    if not has_one:
        score = 1000.0
    else:
        score = -1000.0
        
    # Tiebreaker
    tiebreak = 0
    for i, v in enumerate(vector):
        tiebreak += v * (i + 1)
        
    score += tiebreak * 0.0001
    
    return score
# EVOLVE-BLOCK-END
