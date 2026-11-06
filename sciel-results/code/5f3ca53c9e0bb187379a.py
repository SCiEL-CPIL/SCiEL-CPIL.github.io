import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks
# There are C(16,4) = 1820 such subsets
_ALL_SETS = []
_ALL_SETS_MASKS = []
for combo in combinations(range(16), 4):
    mask = 0
    for bit in combo:
        mask |= (1 << bit)
    _ALL_SETS_MASKS.append(mask)

N_SETS = len(_ALL_SETS_MASKS)  # 1820

# Precompute pairwise intersection sizes
# inter[i][j] = popcount(mask_i & mask_j)
_masks_arr = np.array(_ALL_SETS_MASKS, dtype=np.uint32)

def _popcount(x):
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x000000FF

# Build conflict matrix: conflict[i][j] = True if adding set j to a family containing set i 
# could create a sunflower. Actually, we need a different approach.
# 
# A sunflower of 3 sets A, B, C means |A∩B| = |A∩C| = |B∩C|.
# For 4-sets, possible intersection sizes are 0, 1, 2, 3.
# We need to avoid triples where all three pairwise intersections are equal.
#
# Strategy: Greedy with bitmask optimization and multiple randomized restarts.
# For each candidate set, check if adding it creates a sunflower with any two existing sets.
#
# Precompute for each pair (i,j) the intersection size.
# Then for a new set k, a sunflower exists if there exist i,j in chosen such that
# inter[i][j] == inter[i][k] == inter[j][k].

# Precompute inter matrix as a 2D array
inter = np.zeros((N_SETS, N_SETS), dtype=np.uint8)
for i in range(N_SETS):
    for j in range(i+1, N_SETS):
        v = _popcount(_ALL_SETS_MASKS[i] & _ALL_SETS_MASKS[j])
        inter[i][j] = v
        inter[j][i] = v

def _can_add(chosen_indices, new_idx, inter):
    """Check if adding new_idx to chosen_indices creates a sunflower."""
    n = len(chosen_indices)
    new_inter = inter[new_idx]
    for a in range(n):
        ia = chosen_indices[a]
        ia_new = new_inter[ia]
        for b in range(a+1, n):
            ib = chosen_indices[b]
            if inter[ia][ib] == ia_new and inter[ia][ib] == new_inter[ib]:
                return False
    return True

def _greedy_add(chosen_indices, candidates, inter, rng):
    """Try to add candidates to chosen, return new chosen list."""
    # Try each candidate, add if possible
    for idx in candidates:
        if idx in chosen_indices:
            continue
        if _can_add(chosen_indices, idx, inter):
            chosen_indices.append(idx)
    return chosen_indices

def _greedy_with_order(order, inter, rng):
    """Greedy: process sets in given order, add if possible."""
    chosen = []
    for idx in order:
        if _can_add(chosen, idx, inter):
            chosen.append(idx)
    return chosen

def _local_search(chosen, inter, rng, max_steps=200):
    """Try to improve by swapping out one set for another."""
    chosen_set = set(chosen)
    improved = True
    steps = 0
    while improved and steps < max_steps:
        improved = False
        steps += 1
        # Try removing each element and adding a better one
        for i in range(len(chosen)):
            candidate_remove = chosen[i]
            temp_chosen = chosen[:i] + chosen[i+1:]
            # Try adding each unchosen set
            for j in range(N_SETS):
                if j in chosen_set:
                    continue
                if _can_add(temp_chosen, j, inter):
                    # Check if this is an improvement (it is, since we add one)
                    new_chosen = temp_chosen + [j]
                    # Actually we need to check if the new set is "better" 
                    # For now, just accept any valid swap
                    chosen = new_chosen
                    chosen_set = set(chosen)
                    improved = True
                    break
            if improved:
                break
    return chosen

def _randomized_greedy(inter, rng, deadline):
    """Randomized greedy with shuffled order."""
    order = list(range(N_SETS))
    rng.shuffle(order)
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        if _can_add(chosen, idx, inter):
            chosen.append(idx)
    return chosen

def _greedy_by_degree(inter, rng, deadline):
    """Greedy: pick set with highest "compatibility" first."""
    # Compute for each set how many other sets it can coexist with
    # (i.e., how many sets j such that inter[i][j] != inter[j][k] for all pairs...)
    # This is complex. Simpler: just use random order but weight by something.
    # For now, use a heuristic: sets with fewer conflicts are better.
    # A set i conflicts with pair (j,k) if inter[i][j]==inter[i][k]==inter[j][k]
    # Precompute conflict count for each set
    conflict_count = np.zeros(N_SETS, dtype=np.int32)
    for i in range(N_SETS):
        for j in range(N_SETS):
            if i == j:
                continue
            for k in range(N_SETS):
                if i == k or j == k:
                    continue
                if inter[i][j] == inter[i][k] == inter[j][k]:
                    conflict_count[i] += 1
                    break  # Only count once per (j,k) pair? No, this is wrong.
    # This precomputation is O(N^3) = 1820^3 which is too slow.
    # Skip this and use random restarts instead.
    order = list(range(N_SETS))
    rng.shuffle(order)
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        if _can_add(chosen, idx, inter):
            chosen.append(idx)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    rng = random.Random(42)
    
    best_chosen = []
    best_len = 0
    
    # Multiple randomized restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random order greedy
        order = list(range(N_SETS))
        rng.shuffle(order)
        chosen = []
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            if _can_add(chosen, idx, inter):
                chosen.append(idx)
        
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_chosen = chosen[:]
        
        # Local search on the best so far
        if time.time() > deadline - 10.0:
            break
        
        # Try local search with a few iterations
        improved = True
        ls_steps = 0
        while improved and ls_steps < 50 and time.time() < deadline - 5.0:
            improved = False
            ls_steps += 1
            for i in range(len(chosen)):
                if time.time() > deadline - 5.0:
                    break
                candidate_remove = chosen[i]
                temp_chosen = chosen[:i] + chosen[i+1:]
                # Try adding a random unchosen set
                for _ in range(20):
                    j = rng.randint(0, N_SETS-1)
                    if j in chosen:
                        continue
                    if _can_add(temp_chosen, j, inter):
                        new_chosen = temp_chosen + [j]
                        # Check if this leads to a larger family (it does, size same)
                        # But we want to keep exploring. Accept if valid.
                        chosen = new_chosen
                        improved = True
                        break
                if improved:
                    break
        
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_chosen = chosen[:]
    
    # Convert back to sorted lists
    result = []
    for idx in best_chosen:
        mask = _ALL_SETS_MASKS[idx]
        elems = []
        for bit in range(16):
            if mask & (1 << bit):
                elems.append(bit + 1)
        result.append(sorted(elems))
    
    return result
