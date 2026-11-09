"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
# Precomputation
ALL_TRIPLES = list(itertools.combinations(range(16), 3))
TRIPLE_TO_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES) # 560

# Generate all K(2,2,2) structures
# A K(2,2,2) is defined by 3 disjoint pairs.
# We iterate over all sets of 6 distinct points.
# For each set of 6, there are 15 ways to partition into 3 pairs.
K222_STRUCTURES = []

# Helper to get the 8 transversals given 3 pairs (a,a'), (b,b'), (c,c')
def get_transversals(pairs):
    # pairs = [(a, a'), (b, b'), (c, c')]
    a, ap = pairs[0]
    b, bp = pairs[1]
    c, cp = pairs[2]
    
    # The 8 transversals are combinations of picking one from each pair
    # (a,b,c), (a,b,cp), (a,bp,c), (a,bp,cp), (ap,b,c), (ap,b,cp), (ap,bp,c), (ap,bp,cp)
    transversals = [
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, cp))),
        tuple(sorted((a, bp, c))),
        tuple(sorted((a, bp, cp))),
        tuple(sorted((ap, b, c))),
        tuple(sorted((ap, b, cp))),
        tuple(sorted((ap, bp, c))),
        tuple(sorted((ap, bp, cp)))
    ]
    return transversals

# Precompute pairings of 6 items into 3 pairs
# There are 15 such pairings.
# We can hardcode them or generate them.
# Generating:
def generate_pairings(points):
    # points is a list of 6 elements
    # Fix point 0. Pair it with 1, 2, 3, 4, 5.
    # If paired with 1, remaining 4 have 3 pairings.
    # Total 5 * 3 = 15.
    p = points
    pairings = []
    for i in range(1, 6):
        # Pair p[0] with p[i]
        pair1 = (p[0], p[i])
        remaining = [p[j] for j in range(1, 6) if j != i]
        # Generate pairings for remaining 4
        # Fix remaining[0]. Pair with remaining[1], remaining[2], remaining[3].
        for j in range(1, 4):
            pair2 = (remaining[0], remaining[j])
            rem2 = [remaining[k] for k in range(1, 4) if k != j]
            pair3 = (rem2[0], rem2[1])
            pairings.append([pair1, pair2, pair3])
    return pairings

# Build K222_STRUCTURES
for points in itertools.combinations(range(16), 6):
    points_list = list(points)
    for pairs in generate_pairings(points_list):
        transversals = get_transversals(pairs)
        # Store as tuple of indices
        idxs = tuple(TRIPLE_TO_IDX[t] for t in transversals)
        K222_STRUCTURES.append(idxs)

# Build conflict bitmasks: for each triple, list of masks where each mask has bits
# set for the other 7 triples in a K(2,2,2) containing this triple
conflict_bits = [[] for _ in range(N_TRIPLES)]

for k222 in K222_STRUCTURES:
    for pos in range(8):
        t_idx = k222[pos]
        mask = 0
        for other_pos in range(8):
            if other_pos != pos:
                mask |= (1 << k222[other_pos])
        conflict_bits[t_idx].append(mask)

def _check_addition_fast(triple_idx, S_int):
    """
    Fast check: returns True if adding triple_idx to S_int is valid.
    Pure Python bit operations - no NumPy overhead.
    """
    for mask in conflict_bits[triple_idx]:
        if (S_int & mask) == mask:
            return False
    return True

def _greedy_initial():
    """Generate an initial valid triple set using greedy."""
    indices = list(range(N_TRIPLES))
    random.shuffle(indices)
    
    S_int = 0
    count = 0
    
    for idx in indices:
        if _check_addition_fast(idx, S_int):
            S_int |= (1 << idx)
            count += 1
            
    return S_int, count

def _local_search(S_int, count, time_limit=10.0):
    """Improve the triple set using local search with maintained present/absent lists."""
    start_time = time.time()
    
    # Maintain lists for efficiency
    present_indices = [i for i in range(N_TRIPLES) if (S_int & (1 << i))]
    absent_indices = [i for i in range(N_TRIPLES) if not (S_int & (1 << i))]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try to add any absent triple
        if absent_indices:
            random.shuffle(absent_indices)
            for idx in absent_indices:
                if _check_addition_fast(idx, S_int):
                    S_int |= (1 << idx)
                    count += 1
                    present_indices.append(idx)
                    absent_indices.remove(idx)
                    improved = True
                    break
        
        if not improved:
            # Phase 2: Try 1-remove, 2-add moves (net +1)
            if not present_indices:
                break
                
            random.shuffle(present_indices)
            max_removals = min(20, len(present_indices))
            
            for idx_rem in present_indices[:max_removals]:
                S_int_new = S_int & ~(1 << idx_rem)
                count_new = count - 1
                
                candidates = list(absent_indices)
                random.shuffle(candidates)
                max_adds = min(100, len(candidates))
                
                for idx_add in candidates[:max_adds]:
                    if idx_add == idx_rem:
                        continue
                    if _check_addition_fast(idx_add, S_int_new):
                        S_int_new |= (1 << idx_add)
                        count_new += 1
                        if count_new > count:
                            break
                
                if count_new > count:
                    S_int = S_int_new
                    count = count_new
                    present_indices.remove(idx_rem)
                    for idx_add in candidates[:max_adds]:
                        if idx_add != idx_rem and (S_int & (1 << idx_add)):
                            if idx_add in absent_indices:
                                absent_indices.remove(idx_add)
                                present_indices.append(idx_add)
                    improved = True
                    break
        
        if not improved and len(present_indices) >= 2:
            # Phase 3: Try 2-remove, 3-add moves (net +1)
            random.shuffle(present_indices)
            rem1 = present_indices[0]
            rem2 = present_indices[1]
            if rem1 != rem2:
                S_int_new = S_int & ~(1 << rem1) & ~(1 << rem2)
                count_new = count - 2
                
                candidates = list(absent_indices)
                random.shuffle(candidates)
                max_adds = min(150, len(candidates))
                
                for idx_add in candidates[:max_adds]:
                    if idx_add == rem1 or idx_add == rem2:
                        continue
                    if _check_addition_fast(idx_add, S_int_new):
                        S_int_new |= (1 << idx_add)
                        count_new += 1
                        if count_new > count:
                            break
                
                if count_new > count:
                    S_int = S_int_new
                    count = count_new
                    present_indices.remove(rem1)
                    present_indices.remove(rem2)
                    for idx_add in candidates[:max_adds]:
                        if idx_add not in (rem1, rem2) and (S_int & (1 << idx_add)):
                            if idx_add in absent_indices:
                                absent_indices.remove(idx_add)
                                present_indices.append(idx_add)
                    improved = True
        
        if not improved:
            break
                
    return S_int, count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_S_int = 0
    best_count = 0
    
    while time.time() < deadline:
        S_int, count = _greedy_initial()
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        local_time = min(8.0, remaining - 2.0)
        S_int, count = _local_search(S_int, count, time_limit=local_time)
        
        if count > best_count:
            best_count = count
            best_S_int = S_int
            
    # Convert back to list of triples
    result = []
    for i in range(N_TRIPLES):
        if best_S_int & (1 << i):
            result.append(list(ALL_TRIPLES[i]))
            
    return result
# EVOLVE-BLOCK-END
