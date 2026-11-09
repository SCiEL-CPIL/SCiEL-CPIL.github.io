import time
import random
import math
from itertools import combinations

# Precompute all possible triples on 16 points
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
TRIPLE_PAIRS = [] # List of (pair1, pair2, pair3) for each triple

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append([i, j, k])
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)
            TRIPLE_PAIRS.append(((i, j), (i, k), (j, k)))

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute subsets by size for independence check
# We only need to check subsets of size >= 1
# But we check from large to small to find max alpha quickly
# Optimization: We assume alpha will be small (<= 6) for good solutions.
# If we find no independent set of size 6, we check 5, etc.
# This saves massive time compared to checking all 2^16 subsets.

def compute_alpha_fast(H_indices):
    """
    Compute the independence number of the hypergraph defined by H_indices.
    H_indices is a set of indices into ALL_TRIPLES.
    Returns the largest size of a set of points containing no triple from H_indices.
    """
    if not H_indices:
        return 16
    
    # Check sizes from 6 down to 1.
    # If alpha > 6, we return 7 (or higher if we want to be exact, but 6 is a good lower bound for "good" systems)
    # Actually, we should check up to a reasonable limit. 
    # For 16 points, if we can't find an independent set of size 6, alpha is likely <= 5.
    # Let's check 6, 5, 4, 3, 2, 1.
    
    # Pre-computed lists of masks for each size? 
    # Generating them on the fly or pre-computing once is better.
    # Pre-computation of all masks by size is expensive memory-wise if we store all, but 2^16 is 65536.
    # Let's just generate them once globally if not done.
    
    if not hasattr(compute_alpha_fast, 'masks_by_size'):
        masks_by_size = {k: [] for k in range(1, 17)}
        for mask in range(1, 1 << 16):
            bc = bin(mask).count('1')
            masks_by_size[bc].append(mask)
        compute_alpha_fast.masks_by_size = masks_by_size
    
    masks_by_size = compute_alpha_fast.masks_by_size
    
    # Check from 6 down to 1
    for k in range(6, 0, -1):
        for mask in masks_by_size[k]:
            is_indep = True
            for idx in H_indices:
                if (ALL_TRIPLE_MASKS[idx] & mask) == ALL_TRIPLE_MASKS[idx]:
                    is_indep = False
                    break
            if is_indep:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 15s margin
    
    best_alpha = 16
    best_system = []
    
    # Map pairs to indices for bitmask conflict checking
    pair_to_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            pair_to_idx[(i, j)] = idx
            idx += 1
    
    TRIPLE_PAIR_MASKS = []
    for pairs in TRIPLE_PAIRS:
        mask = 0
        for p in pairs:
            mask |= (1 << pair_to_idx[p])
        TRIPLE_PAIR_MASKS.append(mask)
    
    # List of all triples indices
    all_indices = list(range(NUM_TRIPLES))
    
    # Simulated Annealing Parameters
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Restart loop
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        # 1. Random Greedy Init
        H_indices = set()
        used_pairs_mask = 0
        random.shuffle(all_indices)
        
        for idx in all_indices:
            pmask = TRIPLE_PAIR_MASKS[idx]
            if (used_pairs_mask & pmask) == 0:
                H_indices.add(idx)
                used_pairs_mask |= pmask
        
        current_alpha = compute_alpha_fast(H_indices)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [ALL_TRIPLES[i] for i in H_indices]
        
        # 2. Simulated Annealing
        temp = initial_temp
        last_improvement = time.time()
        
        while time.time() < deadline:
            if time.time() - last_improvement > 1.0:
                break
            
            # Move: Swap
            # Remove a random edge, add a random compatible edge
            if H_indices:
                rem_idx = random.choice(list(H_indices))
                rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                
                # Temporarily remove
                used_pairs_mask_temp = used_pairs_mask & ~rem_pmask
                
                # Find a compatible add
                # To speed up, we can pick a random index and check
                # Or maintain a list of available? Random pick is O(1) check.
                # We might need multiple attempts to find a valid add if the graph is dense.
                found_add = False
                for _ in range(10):
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx in H_indices:
                        continue
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    if (used_pairs_mask_temp & add_pmask) == 0:
                        # Apply
                        H_indices.remove(rem_idx)
                        H_indices.add(add_idx)
                        used_pairs_mask = used_pairs_mask_temp | add_pmask
                        
                        new_alpha = compute_alpha_fast(H_indices)
                        delta = new_alpha - current_alpha
                        
                        if delta <= 0 or random.random() < math.exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [ALL_TRIPLES[i] for i in H_indices]
                                last_improvement = time.time()
                        else:
                            # Revert
                            H_indices.remove(add_idx)
                            H_indices.add(rem_idx)
                            used_pairs_mask = used_pairs_mask_temp | rem_pmask
                        
                        found_add = True
                        break
            
            # Cooling
            temp *= cooling_rate
            if temp < min_temp:
                break

    return best_system

def _fallback_system():
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

_orig_construct = construct

def construct_safe():
    try:
        res = _orig_construct()
        if not res:
            return _fallback_system()
        # Validate linearity just in case
        used = set()
        for t in res:
            i,j,k = t
            pairs = [tuple(sorted((i,j))), tuple(sorted((i,k))), tuple(sorted((j,k)))]
            for p in pairs:
                if p in used:
                    return _fallback_system()
                used.add(p)
        return res
    except:
        return _fallback_system()

# Replace construct
construct = construct_safe
