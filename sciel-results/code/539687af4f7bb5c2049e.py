"""Randomized greedy with bitmask conflict propagation."""
# EVOLVE-BLOCK-START
import itertools
import random

def _precompute():
    """Precompute all 4-sets, intersections, and conflict lookup structures."""
    all_sets = []
    for combo in itertools.combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n = len(all_sets)
    # inter[i][j] = bitmask of intersection of set i and set j
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            k_val = all_sets[i] & all_sets[j]
            inter[i][j] = k_val
            inter[j][i] = k_val
            
    # dicts[i] maps intersection_mask -> bitmask of all sets k such that inter[i][k] == intersection_mask
    dicts = [{} for _ in range(n)]
    for i in range(n):
        d = dicts[i]
        for k in range(n):
            k_val = inter[i][k]
            if k_val in d:
                d[k_val] |= (1 << k)
            else:
                d[k_val] = (1 << k)
                
    return all_sets, inter, dicts

def construct():
    """Construct a large sunflower-free family using randomized greedy with bitmask optimization."""
    all_sets, inter, dicts = _precompute()
    n = len(all_sets)
    
    best_size = 0
    best_indices = []
    
    # Use a fixed seed for reproducibility if needed, but random is better for exploration
    # However, the task requires deterministic output? "Deterministic greedy" was the previous title.
    # The task contract doesn't strictly forbid randomness, but "Deterministic" in the previous title suggests it.
    # But "Maximize family cardinality" often benefits from randomness.
    # I will use a fixed seed for the random shuffles to ensure reproducibility.
    rng = random.Random(42)
    
    num_trials = 200 # Should be fast enough
    
    for _ in range(num_trials):
        order = list(range(n))
        rng.shuffle(order)
        
        chosen = []
        bad = 0 # Bitmask of indices that form a sunflower with some pair in chosen
        
        for idx in order:
            # Check if this index is already blocked
            if (bad >> idx) & 1:
                continue
            
            # Add idx to chosen
            chosen.append(idx)
            
            # Update bad bitmask
            # For every existing j in chosen (before adding idx, or including? 
            # If we include idx, inter[idx][idx] = all_sets[idx].
            # K = all_sets[idx] & all_sets[idx] = all_sets[idx].
            # dicts[idx][all_sets[idx]] contains idx.
            # dicts[j][all_sets[idx]] ...
            # We need pairs (idx, j) for j in chosen_old.
            # So we iterate over chosen[:-1]
            
            for j in chosen[:-1]:
                K = inter[idx][j]
                # Get all k such that inter[idx][k] == K and inter[j][k] == K
                mask_k = dicts[idx][K] & dicts[j][K]
                bad |= mask_k
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_indices = chosen[:]
            
    # Convert best_indices back to sorted lists of elements
    result = []
    for idx in best_indices:
        mask = all_sets[idx]
        # Extract elements
        elements = []
        for i in range(16):
            if (mask >> i) & 1:
                elements.append(i + 1)
        result.append(elements)
        
    return result
# EVOLVE-BLOCK-END
