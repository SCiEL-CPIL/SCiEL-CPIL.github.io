import time
import random
import math
import sys
from itertools import combinations

# Precompute all possible triples on 16 points
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
TRIPLE_PAIRS = []

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append((i, j, k))
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)
            TRIPLE_PAIRS.append(((i, j), (i, k), (j, k)))

NUM_TRIPLES = len(ALL_TRIPLES)

# Map pairs to indices for bitmask representation
# There are C(16,2) = 120 pairs
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

# Precompute subsets by size for independence check
# We check from large to small to find max alpha quickly
# To save memory and time, we might only need to check down to a certain size
# But for correctness, we check all.
# Optimization: Instead of storing all masks, we can generate them on the fly?
# No, precomputation is fine for 16 points. 2^16 = 65536 subsets.
SUBSETS_BY_SIZE = {k: [] for k in range(1, 17)}
for mask in range(1, 1 << 16):
    bc = mask.bit_count()
    if bc <= 16:
        SUBSETS_BY_SIZE[bc].append(mask)

def is_independent(mask, H_indices):
    """
    Check if the set of vertices represented by mask is independent
    given the list of triple indices H_indices.
    """
    for idx in H_indices:
        if (ALL_TRIPLE_MASKS[idx] & mask) == ALL_TRIPLE_MASKS[idx]:
            return False
    return True

def compute_alpha(H_indices):
    """
    Compute the independence number of the hypergraph defined by H_indices.
    H_indices is a set of indices into ALL_TRIPLES.
    """
    if not H_indices:
        return 16
    
    H_list = list(H_indices)
    
    # Check from largest possible size down
    for k in range(16, 0, -1):
        for mask in SUBSETS_BY_SIZE[k]:
            if is_independent(mask, H_list):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin from 120s limit
    
    best_alpha = 16
    best_system = []
    
    # Strategy:
    # 1. Random Greedy Packing.
    # 2. Simulated Annealing with bitmasks for pair conflicts.
    # 3. Focus on swaps to maintain size and improve structure.
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # 1. Random Greedy Packing
        H_indices = set()
        used_pairs_mask = 0
        
        available = list(range(NUM_TRIPLES))
        random.shuffle(available)
        
        for idx in available:
            pmask = TRIPLE_PAIR_MASKS[idx]
            if (used_pairs_mask & pmask) == 0:
                H_indices.add(idx)
                used_pairs_mask |= pmask
        
        current_alpha = compute_alpha(H_indices)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [list(ALL_TRIPLES[i]) for i in H_indices]
        
        # 2. Simulated Annealing
        temp = 1.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        # Precompute list of H indices for fast choice?
        # We'll use list(H_indices) for random.choice
        
        while temp > min_temp and time.time() < deadline:
            # Choose a move type
            # Focus on swaps (remove one, add one) as they are most likely to improve structure
            # Occasionally add or remove if possible
            
            move_type = random.random()
            
            if move_type < 0.8:
                # Swap
                if not H_indices:
                    continue
                    
                rem_idx = random.choice(list(H_indices))
                rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                
                # Temporarily remove
                used_pairs_mask_temp = used_pairs_mask & ~rem_pmask
                
                # Find a candidate to add
                # Try a few random ones to find a valid one quickly
                found_add = False
                for _ in range(20):
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx in H_indices:
                        continue
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    if (used_pairs_mask_temp & add_pmask) == 0:
                        # Valid move
                        H_indices.remove(rem_idx)
                        H_indices.add(add_idx)
                        used_pairs_mask = used_pairs_mask_temp | add_pmask
                        
                        new_alpha = compute_alpha(H_indices)
                        delta = new_alpha - current_alpha
                        
                        # Accept if better or with probability exp(-delta/temp)
                        if delta <= 0 or random.random() < math.exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [list(ALL_TRIPLES[i]) for i in H_indices]
                        else:
                            # Revert
                            H_indices.remove(add_idx)
                            H_indices.add(rem_idx)
                            used_pairs_mask = used_pairs_mask_temp | rem_pmask
                        
                        found_add = True
                        break
                
                if not found_add:
                    # If we couldn't find a valid add, do nothing (state unchanged)
                    pass
                    
            elif move_type < 0.9:
                # Add
                # Pick random idx not in H
                for _ in range(20):
                    idx = random.randint(0, NUM_TRIPLES - 1)
                    if idx in H_indices:
                        continue
                    pmask = TRIPLE_PAIR_MASKS[idx]
                    if (used_pairs_mask & pmask) == 0:
                        # Valid move
                        H_indices.add(idx)
                        used_pairs_mask |= pmask
                        
                        new_alpha = compute_alpha(H_indices)
                        delta = new_alpha - current_alpha
                        
                        if delta <= 0 or random.random() < math.exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [list(ALL_TRIPLES[i]) for i in H_indices]
                        else:
                            # Revert
                            H_indices.remove(idx)
                            used_pairs_mask &= ~pmask
                        break

            else:
                # Remove
                if H_indices:
                    rem_idx = random.choice(list(H_indices))
                    rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                    
                    H_indices.remove(rem_idx)
                    used_pairs_mask &= ~rem_pmask
                    
                    new_alpha = compute_alpha(H_indices)
                    delta = new_alpha - current_alpha
                    
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [list(ALL_TRIPLES[i]) for i in H_indices]
                    else:
                        # Revert
                        H_indices.add(rem_idx)
                        used_pairs_mask |= rem_pmask
            
            temp *= cooling_rate
            
            # Check time
            if time.time() > deadline:
                break
        
        # End SA for this restart

    return best_system

def _fallback_system():
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

def construct_safe():
    try:
        res = construct()
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

construct = construct_safe
