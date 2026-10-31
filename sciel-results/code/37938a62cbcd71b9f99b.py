import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 2.0

# We will maintain a small "core" of accepted vectors to measure distance against.
# Since we don't know the true accepted set (the evaluator hides it), we approximate
# by tracking vectors that we *predict* will be accepted.
# A vector is predicted accepted if it has a high priority AND is far from our core.
# However, a simpler and robust proxy: track the top-K vectors by our own priority,
# and assume they are likely in the set. Then, for new vectors, prioritize those
# that are far from this core.

# To make this efficient, we'll keep a list of "anchors".
# If a vector is very far from all anchors, it's a good candidate.
# If it's close, it's likely to form an AP or be redundant, so penalize it.

# We need to be careful: the evaluator calls priority() in sorted order.
# So we don't see the "accepted" status directly.
# But we can infer: if a vector has high priority, it's likely accepted.
# So we can update our "core" with high-priority vectors.

# Let's maintain a list of core vectors.
# We'll add a vector to the core if its priority is high enough.
# What is "high enough"? We don't know the scale.
# Let's assume the top 10-20% of vectors seen so far are "good".
# Or, simpler: just keep the last N vectors that had the highest priority values.

# Actually, the most robust "state-aware" signal without knowing acceptance is:
# The density of high-priority vectors in the neighborhood.
# But that's hard to compute online.

# Alternative structural change:
# Instead of static features, use a *random projection* based score that changes
# based on the *order* of calls? No, that's not state-aware of the set.

# Let's try the "dynamic core" approach.
# We maintain a list `core` of vectors.
# When `priority` is called:
# 1. Compute a base score (e.g., balance).
# 2. If the base score is very high (top quantile), add to core.
# 3. Compute distance to core.
# 4. Final score = base_score - penalty * (min distance to core).

# Wait, if we penalize closeness to core, we are doing a greedy packing.
# This is a fundamentally different strategy from static balancing.

# Optimization:
# Computing distance to all core vectors is O(K*n).
# If K is small (e.g., 100), and n is small (e.g., 8-10), this is fast.
# If n is large (e.g., 20+), K*n might be slow if called millions of times.
# We need to limit the size of `core` and the number of distance checks.

# Let's cap `core` size at, say, 50.
# And we'll only do the distance check if we have enough time.

_core = []
_CORE_MAX_SIZE = 50
# We need a threshold to decide when to add to core.
# Let's keep track of the max priority seen so far.
_max_priority_seen = -1e18
# And a running quantile? Too complex.
# Let's just add to core if priority > some threshold.
# Since we don't know the scale, let's use a relative threshold:
# If priority is in the top 1% of all seen? Hard to do online without storing all.
# Let's just add the top _CORE_MAX_SIZE vectors by priority.
# We can maintain a min-heap of size _CORE_MAX_SIZE.

import heapq

_core_heap = []  # min-heap of (priority, vector)

def _time_check():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def _hamming_dist(v1, v2):
    # v1, v2 are tuples of ints
    d = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            d += 1
    return d

def priority(vector, n):
    # 1. Compute base score (balance)
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
    
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -deviation * 10000.0
    
    # Add small hash for tie-breaking
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    hash_score = h / (10**9 + 7)
    
    base_final = balance_score + hash_score

    # 2. State-aware adjustment
    if _time_check():
        return base_final

    # If we have a core, compute min distance to core
    if _core_heap:
        # To avoid O(K*n) cost for every call, we can sample or limit.
        # But K is small (50). n is likely small.
        # Let's compute min distance.
        min_dist = float('inf')
        # We need to access the vectors in the heap.
        # The heap stores (priority, vector).
        # We can iterate through the heap to find min distance.
        # This is O(K).
        for p, vec in _core_heap:
            d = _hamming_dist(vector, vec)
            if d < min_dist:
                min_dist = d
        
        # Penalize closeness.
        # If min_dist is small, the vector is "crowded".
        # We want to spread out.
        # Penalty = max(0, threshold - min_dist) * weight
        # What is a good threshold?
        # In {0,1,2}^n, max distance is n.
        # Average random distance is n/3.
        # Let's penalize if dist < n/3.
        threshold = n / 3.0
        if min_dist < threshold:
            penalty = (threshold - min_dist) * 100.0
        else:
            penalty = 0.0
        
        base_final -= penalty

    # 3. Update core
    # If this vector is "good" (high base score), add to core.
    # How to decide "good"?
    # Let's add if base_final is higher than the minimum in the heap.
    if len(_core_heap) < _CORE_MAX_SIZE:
        heapq.heappush(_core_heap, (base_final, vector))
    else:
        # If heap is full, compare with the smallest
        if base_final > _core_heap[0][0]:
            heapq.heapreplace(_core_heap, (base_final, vector))

    return float(base_final)
