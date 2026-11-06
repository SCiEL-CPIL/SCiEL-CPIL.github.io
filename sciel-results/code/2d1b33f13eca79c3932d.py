import time
import math
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 55.0  # Safety margin

N = 24

def _time_left():
    return _DEADLINE - (time.time() - _START)

# Precompute LCM and weights
_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [_L // k for k in range(1, 25)]

# Precompute all subset sums for each target
# _subset_sums[target] = set of all possible subset sums of weights from [1..24]\{target}
# But we need to be smarter: for the conflict graph, we need to know if 1/i is a sum
# of reciprocals of a subset of [1..24]\{i,j} for each pair (i,j)

# Actually, let's precompute for each element i, the set of all subset sums of weights
# from elements other than i. Then for pair (i,j), we check if _W[i-1] is in 
# subset_sums_excluding_i_and_j.

# Let's precompute subset_sums[i] = set of all subset sums of weights from [1..24]\{i+1}
# (using 0-indexed: element i+1 is excluded)

_subset_sums_excluding = [None] * 24

for excl in range(24):
    # Compute all subset sums of weights from all elements except excl
    items = [_W[k] for k in range(24) if k != excl]
    sums = {0}
    for item in items:
        new_sums = set()
        for s in sums:
            ns = s + item
            if ns <= _W[excl]:  # We only care about sums up to the target
                new_sums.add(ns)
        sums |= new_sums
    _subset_sums_excluding[excl] = sums

# Now build the conflict graph
# conflict[i][j] = True if 1/(i+1) is a sum of reciprocals of a subset of [1..24]\{i+1, j+1}
# i.e., _W[i] is in subset_sums_excluding[i] but NOT achievable using only elements other than i and j
# Wait, that's not quite right. Let me think again.

# For pair (i,j) (0-indexed, representing elements i+1 and j+1):
# We need to check if _W[i] (weight of element i+1) equals sum of weights of some subset
# of {1..24} \ {i+1, j+1}.
# So we need subset sums of weights from all elements except i+1 and j+1.

# Let's precompute subset_sums_excluding_pair[i][j] for all pairs
# This is 24*23/2 = 276 pairs, each requiring subset sum computation.

# Actually, let's be smarter. For each element i, we have _subset_sums_excluding[i] which
# includes sums using any subset of [1..24]\{i+1}. For pair (i,j), we need sums that don't
# use element j+1 either. So we need to check if _W[i] can be formed without using element j+1.

# Let's precompute for each i and each potential "forbidden" element j, whether _W[i]
# is achievable using elements from [1..24]\{i+1, j+1}.

# This is still O(24*24*2^22) which is too much. Let's use a different approach.

# For the conflict graph, we only need to know if there EXISTS a subset of [1..24]\{i+1, j+1}
# that sums to _W[i]. We can use meet-in-the-middle for each pair.

def _can_form_target_without(target_weight, exclude1, exclude2):
    """Check if target_weight can be formed as sum of subset of _W excluding exclude1 and exclude2 (0-indexed)"""
    items = [_W[k] for k in range(24) if k != exclude1 and k != exclude2]
    
    if not items:
        return target_weight == 0
    
    # Meet in the middle
    half = len(items) // 2
    left_items = items[:half]
    right_items = items[half:]
    
    left_sums = {0}
    for item in left_items:
        new_sums = set()
        for s in left_sums:
            ns = s + item
            if ns <= target_weight:
                new_sums.add(ns)
        left_sums |= new_sums
    
    right_sums = {0}
    for item in right_items:
        new_sums = set()
        for s in right_sums:
            ns = s + item
            if ns <= target_weight:
                new_sums.add(ns)
        right_sums |= new_sums
    
    for ls in left_sums:
        if (target_weight - ls) in right_sums:
            return True
    return False

# Precompute conflict graph
# conflict[i][j] = True if elements (i+1) and (j+1) cannot both be in a valid set
# i.e., 1/(i+1) = sum of reciprocals of some subset of [1..24]\{i+1, j+1}
# OR 1/(j+1) = sum of reciprocals of some subset of [1..24]\{i+1, j+1}

print("Computing conflict graph...")
_conflict = [[False] * 24 for _ in range(24)]

for i in range(24):
    for j in range(i+1, 24):
        # Check if 1/(i+1) is sum of subset of [1..24]\{i+1, j+1}
        if _can_form_target_without(_W[i], i, j):
            _conflict[i][j] = True
            _conflict[j][i] = True
            continue
        # Check if 1/(j+1) is sum of subset of [1..24]\{i+1, j+1}
        if _can_form_target_without(_W[j], i, j):
            _conflict[i][j] = True
            _conflict[j][i] = True

print("Conflict graph computed. Edges:", sum(sum(row) for row in _conflict) // 2)

# Now find Maximum Independent Set using branch and bound
# Use bitsets for efficiency

def _max_independent_set():
    """Find maximum independent set in the conflict graph using branch and bound"""
    n = 24
    
    # Precompute bitsets for adjacency
    adj_bits = [0] * n
    for i in range(n):
        for j in range(n):
            if _conflict[i][j]:
                adj_bits[i] |= (1 << j)
    
    # Also precompute non-adjacency bits (including self)
    non_adj_bits = [0] * n
    for i in range(n):
        non_adj_bits[i] = ((1 << n) - 1) ^ adj_bits[i]
        non_adj_bits[i] |= (1 << i)  # Include self
    
    best_size = 0
    best_set = []
    
    def _branch(candidates_bits, current_set, current_size):
        nonlocal best_size, best_set
        
        if _time_left() < 2.0:
            return
        
        # Upper bound: current_size + popcount(candidates_bits)
        if current_size + bin(candidates_bits).count('1') <= best_size:
            return
        
        if candidates_bits == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Select a vertex to branch on
        # Use a heuristic: pick vertex with fewest candidates (degree in induced subgraph)
        v = -1
        min_degree = n + 1
        temp_bits = candidates_bits
        while temp_bits:
            lsb = temp_bits & (-temp_bits)
            idx = lsb.bit_length() - 1
            # Count candidates adjacent to idx within candidates
            degree = bin(candidates_bits & adj_bits[idx]).count('1')
            if degree < min_degree:
                min_degree = degree
                v = idx
            temp_bits ^= lsb
        
        if v == -1:
            return
        
        # Branch 1: Include v
        new_candidates = candidates_bits & non_adj_bits[v] & ~(1 << v)
        current_set.append(v)
        _branch(new_candidates, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude v
        _branch(candidates_bits & ~(1 << v), current_set, current_size)
    
    all_bits = (1 << n) - 1
    _branch(all_bits, [], 0)
    
    return best_set

# Convert to 1-indexed and validate
def _validate_set(s):
    """Check if set s (1-indexed) is valid"""
    s_set = set(s)
    for x in s_set:
        target_w = _W[x - 1]
        others = [y for y in s_set if y != x]
        other_w = [_W[y - 1] for y in others]
        
        # Meet in the middle
        if not other_w:
            continue
        half = len(other_w) // 2
        left_items = other_w[:half]
        right_items = other_w[half:]
        
        left_sums = {0}
        for item in left_items:
            new_sums = set()
            for sum_val in left_sums:
                ns = sum_val + item
                if ns <= target_w:
                    new_sums.add(ns)
            left_sums |= new_sums
        
        right_sums = {0}
        for item in right_items:
            new_sums = set()
            for sum_val in right_sums:
                ns = sum_val + item
                if ns <= target_w:
                    new_sums.add(ns)
            right_sums |= new_sums
        
        for ls in left_sums:
            if (target_w - ls) in right_sums:
                return False
    return True

def construct():
    # Find maximum independent set
    mis = _max_independent_set()
    
    # Convert to 1-indexed
    result = [x + 1 for x in mis]
    
    # Validate
    if not result or not _validate_set(result):
        # Fallback: use greedy
        result = []
        for elem in range(24, 0, -1):
            if _validate_set(result + [elem]):
                result.append(elem)
    
    return result
