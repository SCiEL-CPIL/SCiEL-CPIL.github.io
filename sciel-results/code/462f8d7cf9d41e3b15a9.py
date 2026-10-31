import time
import math

# Global state to track the "crowding" of values in each coordinate
# This allows the priority function to be adaptive to the current set composition
# without needing to re-scan the entire set (which we don't have access to directly,
# but we can approximate the "pressure" by tracking how many times we've seen
# each (coord, value) pair in the vectors that were *accepted* or *considered*).
# 
# Since we don't know which vectors are accepted until after the sort, 
# a true stateful approach is hard in a single-pass sort. 
# However, the greedy algorithm adds vectors in descending priority.
# We can simulate the "ideal" set structure.
#
# A better approach for a static priority function that mimics statefulness:
# Use a hash-based projection that correlates with "orthogonality" or "spread".
#
# Let's try a different angle: 
# The current program scores based on symbol balance (0,1,2 counts).
# A cap set cannot contain any 3 distinct points x, y, z such that x+y+z = 0 mod 3.
# Equivalently, no 3 points form an affine line.
#
# Heuristic: Vectors that are "far apart" in some metric are better.
# But we can't compute pairwise distances in O(N) per vector.
#
# Alternative Strategy: 
# The previous program used L1 deviation from balance.
# Let's try a **frequency-based penalty** on pairs of coordinates.
# If two coordinates (i, j) have the same value in many high-priority vectors,
# it creates a risk of lines.
#
# Actually, the most effective "static" proxy for cap-set-likeness is often
# related to the **entropy** or **distinctness** of the vector.
#
# Let's try a **Bitwise/Hash-based diversity** score.
# We want vectors that are "different" from each other.
# Since we can't see the set, we can use a property of the vector itself.
#
# New Strategy: **Coordinate Independence Score**
# Count how many pairs of coordinates (i, j) have the same value.
# A vector with all distinct values (if n<=3) is "good".
# For larger n, we want to minimize the "redundancy" of values.
#
# Let's compute the number of "collisions" in the vector.
# Collision: v[i] == v[j] for i < j.
# We want to MINIMIZE collisions.
#
# Additionally, we can look at the "distance" from the center in a Hamming-like sense.
#
# Let's combine:
# 1. Balance score (as before, but maybe weighted differently)
# 2. Collision penalty (minimize pairs of equal values)
# 3. A strong hash tie-breaker to ensure deterministic but "random-looking" order for ties.

_start_time = time.time()
_DEADLINE = _start_time + 280.0  # 50s safety margin

# Precompute powers or bases? No, simple loop is fine.

def priority(vector, n):
    # Deadline check
    if time.time() > _DEADLINE:
        # Return a very low priority to stop processing? 
        # No, the evaluator sorts once. We just need to be fast.
        # If we are over time, just return 0.0 to be safe.
        return 0.0

    # 1. Count values
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

    # 2. Balance Score (Minimize deviation from n/3)
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # 3. Collision Score (Minimize pairs of equal values)
    # Total pairs = n*(n-1)/2
    # Pairs of 0s = c0*(c0-1)/2
    # Pairs of 1s = c1*(c1-1)/2
    # Pairs of 2s = c2*(c2-1)/2
    collisions = (c0*(c0-1) + c1*(c1-1) + c2*(c2-1)) * 0.5
    
    # We want to MINIMIZE collisions.
    # Scale it to be comparable to deviation.
    # Max deviation is roughly n. Max collisions is n^2/6.
    # For n=10, max dev ~ 10, max coll ~ 15.
    # Let's weight collisions heavily as they directly imply structure that might form lines.
    
    score = -deviation * 100.0 - collisions * 10.0

    # 4. Tie-breaker: Deterministic Hash
    # Use a faster hash than polynomial if possible, or just a simple one.
    # Python's hash() is randomized per run, so we need a deterministic one.
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    
    # Add a tiny fraction to break ties uniquely
    score += h / 4294967296.0

    return float(score)
