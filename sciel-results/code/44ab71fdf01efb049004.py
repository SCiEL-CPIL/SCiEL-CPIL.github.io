import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16] as bitmasks for fast intersection checks
# Index i corresponds to the i-th combination
ALL_COMBOS = list(combinations(range(1, 17), 4))
NUM_SETS = len(ALL_COMBOS)

# Convert to bitmasks
ALL_MASKS = [0] * NUM_SETS
for i, c in enumerate(ALL_COMBOS):
    m = 0
    for bit in c:
        m |= (1 << (bit - 1))
    ALL_MASKS[i] = m

# Precompute pairwise intersections (as bitmasks) and whether they form a sunflower with a third set
# A sunflower of 3 sets S1, S2, S3 exists if (S1 & S2) == (S1 & S3) == (S2 & S3)
# Let I12 = S1 & S2, I13 = S1 & S3, I23 = S2 & S3.
# The condition is I12 == I13 == I23.
# This is equivalent to:
# 1. I12 == I13  => S1 & S2 == S1 & S3
# 2. I12 == I23  => S1 & S2 == S2 & S3
#
# Note: I12 == I13 implies (S1 & S2) & ~S3 == 0 and (S1 & S3) & ~S2 == 0?
# Actually, simpler:
# I12 == I13 <=> (S1 & S2 & ~S3) is empty and (S1 & S3 & ~S2) is empty.
# I12 == I23 <=> (S1 & S2 & ~S3) is empty and (S2 & S3 & ~S1) is empty.
#
# Let's precompute for every pair (i, j), the intersection mask.
# And for a new set k, to check if it conflicts with any pair (i, j) in the family,
# we need to check if (ALL_MASKS[i] & ALL_MASKS[k]) == (ALL_MASKS[j] & ALL_MASKS[k]) == (ALL_MASKS[i] & ALL_MASKS[j]).
#
# A more efficient way to check conflict when adding S_k to family F:
# For every pair (i, j) in F, let I_ij = S_i & S_j.
# The new set S_k creates a sunflower with i and j if:
# (S_k & S_i) == (S_k & S_j) == I_ij.
#
# We can precompute for each pair (i, j) the intersection I_ij.
# Then, for a candidate k, we need to check if there exists a pair (i, j) in F such that:
# (S_k & S_i) == I_ij AND (S_k & S_j) == I_ij.
#
# This is still O(|F|^2) per candidate.
#
# Alternative structural change:
# Use a different representation or search strategy.
# Since n=16, k=4, the total number of sets is 1820.
# We can use a bitset-based approach to track which pairs are "blocked" or use a different conflict detection.
#
# Let's try a different angle: Local Search / Hill Climbing.
# Start with a random valid family (greedy) and then try to swap sets or add/remove to improve.
# Or, use a SAT-like constraint propagation?
#
# Let's stick to optimizing the greedy but with a different heuristic for ordering.
# Instead of random shuffle, use a heuristic that prioritizes sets with fewer conflicts.
# Or, use "simulated annealing" style moves.
#
# Actually, the current greedy is quite slow because of the O(N^3) conflict check (N candidates * |F|^2 checks).
# 1820 candidates * (avg |F|^2). If |F| ~ 100, that's 1820 * 10000 = 18M ops per pass.
# With 200 passes, that's 3.6B ops. Too slow for Python.
#
# We need a faster conflict check or a smarter search.
#
# Insight:
# If we maintain a data structure that allows us to quickly check if adding S_k causes a conflict.
# A conflict occurs if there exist i, j in F such that:
# (S_k & S_i) == (S_k & S_j) == (S_i & S_j).
#
# Let's precompute I_ij for all pairs (i, j).
# For a fixed k, the condition is:
# For some i, j in F:
# I_ki == I_kj == I_ij.
#
# This is hard to index directly.
#
# However, note that I_ki is a subset of S_k.
# There are only 2^4 = 16 possible subsets of S_k.
# So, for a fixed k, we can group the existing family members F by their intersection with S_k.
# Let Bucket[x] = { i in F | S_i & S_k == x }.
# A conflict exists if there is a bucket x with size >= 2, and for any pair i, j in Bucket[x],
# we have I_ij == x.
#
# So, for each k, we:
# 1. Compute intersection of S_k with all i in F. (O(|F|))
# 2. Group them into buckets by intersection value. (O(|F|))
# 3. For each bucket with size >= 2, check if any pair (i, j) in the bucket has I_ij == x.
#    Since the bucket is defined by S_i & S_k == x, we just need to check if I_ij == x.
#    We can precompute I_ij for all pairs.
#
# How many pairs do we check in step 3?
# In the worst case, one bucket has size |F|. Then we check |F|^2 / 2 pairs.
# But on average, the buckets are smaller.
# Also, we can optimize: If a bucket has size > 1, we only need to find ONE pair that conflicts.
#
# Let's implement this optimized greedy.
# Precompute I_ij matrix? 1820x1820 is ~3.3M entries. That's fine.
# But we don't need the full matrix, just the value I_ij for pairs in F.
# We can compute I_ij on the fly using bitmasks: ALL_MASKS[i] & ALL_MASKS[j].
#
# Let's rewrite the greedy loop with this optimization.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_family = []
    best_size = 0
    
    # Use multiple random seeds for diversity
    random.seed(42) # Reproducibility for first run, but we can vary this
    
    # Precompute all masks
    # ALL_MASKS is already global
    
    # We will run multiple greedy passes.
    # To speed up, we can use a "conflict count" heuristic for ordering.
    # But let's first make the basic greedy faster.
    
    for attempt in range(500): # More attempts if faster
        if time.time() > deadline - 2.0:
            break
            
        # Shuffle indices
        indices = list(range(NUM_SETS))
        random.shuffle(indices)
        
        family = []
        family_set = set() # For fast membership check if needed, but we use list for iteration
        
        # For each candidate, check conflict
        for k in indices:
            mask_k = ALL_MASKS[k]
            
            # Check if k is already in family (shouldn't happen with unique indices, but safe)
            if k in family_set:
                continue
                
            # Check conflict with existing family
            conflict = False
            
            # Bucket family members by intersection with k
            buckets = {}
            for i in family:
                inter = mask_k & ALL_MASKS[i]
                if inter not in buckets:
                    buckets[inter] = []
                buckets[inter].append(i)
                
            # Check each bucket
            for inter_val, members in buckets.items():
                m_len = len(members)
                if m_len < 2:
                    continue
                
                # Check if any pair in this bucket has intersection == inter_val
                # members is a list of indices
                for a_idx in range(m_len):
                    i = members[a_idx]
                    mask_i = ALL_MASKS[i]
                    for b_idx in range(a_idx + 1, m_len):
                        j = members[b_idx]
                        mask_j = ALL_MASKS[j]
                        # Check if I_ij == inter_val
                        if (mask_i & mask_j) == inter_val:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(k)
                family_set.add(k)
                
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = family[:]
                    
    # Convert best family indices back to sorted lists of elements
    result = []
    for idx in best_family:
        result.append(list(ALL_COMBOS[idx]))
        
    return result
