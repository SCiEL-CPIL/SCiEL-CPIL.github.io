"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Count occurrences of 0, 1, 2
    c0 = 0
    c1 = 0
    c2 = 0
    for x in vector:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Heuristic:
    # 1. Penalize 1s (midpoint risk). Fewer 1s is better.
    # 2. Penalize imbalance between 0s and 2s. Balanced is better.
    # 3. Tie-breaker: Prefer lower sum (arbitrary but consistent)
    
    score = -c1 - 0.5 * abs(c0 - c2)
    
    # Add a tiny tie-breaker to ensure deterministic ordering for equal structural scores
    # using the lexicographic value of the vector
    # This prevents reliance on input order for ties
    val = 0
    base = 1
    for x in reversed(vector):
        val += x * base
        base *= 3
    
    # Subtract val to prioritize lower lexicographic values (arbitrary choice)
    # Scale down to not dominate the main structural score
    score -= val * 1e-6
    
    return score
# EVOLVE-BLOCK-END
