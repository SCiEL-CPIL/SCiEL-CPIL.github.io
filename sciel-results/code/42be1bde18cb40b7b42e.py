# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    Higher priority = added earlier.
    
    Strategy: In {0,1,2}^n, a 3-term AP {a,b,c} requires a+c=2b component-wise.
    The value 1 is the unique midpoint (0+2=2*1). Vectors with fewer 1s are
    less likely to serve as midpoints in APs, so they are "safer" to add early.
    We also prefer vectors that are more "extreme" (closer to all-0s or all-2s)
    as they have fewer mixed coordinates that could form APs.
    
    Primary signal: minimize count of 1s (fewer midpoints)
    Secondary signal: maximize variance of coordinates (more extreme vectors)
    Tertiary: deterministic tiebreaker
    """
    c1 = vector.count(1)
    
    # Primary: fewer 1s is better (1 is the only possible midpoint value)
    score = -c1 * 100.0
    
    # Secondary: prefer vectors with more extreme values
    # Compute sum of squared deviations from 1 (variance-like measure)
    # Higher variance = more 0s and 2s, fewer 1s = safer
    var_sum = sum((v - 1) ** 2 for v in vector)
    score += var_sum * 10.0
    
    # Tertiary: deterministic tiebreaker using weighted position sum
    tiebreak = 0
    for i, v in enumerate(vector):
        tiebreak += v * (i + 1)
    score += tiebreak * 0.001
    
    return score
# EVOLVE-BLOCK-END


def _main():
    """
    Main entry point called by the evaluator.
    Determines n from context or uses a reasonable default.
    """
    import time
    start_time = time.time()
    
    # The task says vectors of {0,1,2}^n, but n is not directly provided
    # In the FunSearch contract, n is passed to priority()
    # We need to figure out what n to use. Looking at the task, it seems
    # n should be a parameter. Let me re-read...
    # 
    # The interface is: priority(vector: tuple, n: int) -> float
    # So the evaluator calls priority with a specific n.
    # But we also need to return the cap set size.
    # 
    # Looking at the seed program, it just returns 0.0 from priority.
    # The score is "cap set size" and the evaluator measures the raw value.
    # 
    # I think the evaluator:
    # 1. Calls priority() for each vector in some order
    # 2. Sorts vectors by priority descending
    # 3. Greedily adds vectors that maintain cap set property
    # 4. Returns the final cap set size as the score
    #
    # So my priority function just needs to return good priorities.
    # The actual greedy selection is done by the evaluator!
    #
    # Wait, but the task says "vectors of {0,1,2}^n are added greedily in 
    # descending priority when they keep the set a cap set"
    #
    # So the evaluator does the greedy selection. My job is just to provide
    # good priority values.
    #
    # But then how do I return the cap set size? The priority function returns
    # a float priority, not the cap set size.
    #
    # Let me re-read: "Score: cap set size" and "The evaluator measures the 
    # raw value"
    #
    # I think the evaluator:
    # 1. Generates all vectors in {0,1,2}^n for some n
    # 2. Calls priority(vector, n) for each
    # 3. Sorts by priority descending
    # 4. Greedily adds to cap set
    # 5. Returns len(cap_set) as the score
    #
    # So I just need a good priority function. The current one returns 0.0 
    # which gives lexicographic order (or some default order).
    #
    # My improved priority should favor balanced vectors.
    pass


# The actual entry point that the evaluator calls
# Based on the task description, the evaluator calls priority() and does
# the greedy selection itself. So I just need to provide a good priority.

def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    Higher priority = added earlier.
    
    Strategy: Favor vectors with balanced symbol counts (close to n/3 each).
    This avoids early saturation of specific patterns and reduces the
    likelihood of forming 3-term arithmetic progressions.
    """
    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal: each symbol appears n/3 times
    ideal = n / 3.0
    
    # Deviation from ideal balance
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary score: minimize deviation
    score = -deviation * 1000.0
    
    # Secondary: deterministic tiebreaker to ensure consistent ordering
    # Use a weighted sum based on position
    tiebreak = 0.0
    for i, v in enumerate(vector):
        # Weight by position to break ties deterministically
        tiebreak += v * (i + 1)
    score += tiebreak * 0.001
    
    return score
# EVOLVE-BLOCK-END
