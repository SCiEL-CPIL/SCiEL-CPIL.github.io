import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 16
    k = 4
    all_sets = []
    mask_to_idx = {}
    mask_to_bits = {}
    
    for comb in combinations(range(1, n+1), k):
        mask = 0
        for x in comb:
            mask |= (1 << (x-1))
        idx = len(all_sets)
        all_sets.append(mask)
        mask_to_idx[mask] = idx
        mask_to_bits[mask] = list(comb)
        
    num_sets = len(all_sets)
    
    # Precompute conflicts in a 2D list
    # conf[i][j] is a list of indices k
    conf = [[[] for _ in range(num_sets)] for __ in range(num_sets)]
    
    universe = (1 << n) - 1
    
    for i in range(num_sets):
        mask_i = all_sets[i]
        bits_i = mask_to_bits[mask_i]
        for j in range(i+1, num_sets):
            mask_j = all_sets[j]
            bits_j = mask_to_bits[mask_j]
            
            inter_mask = mask_i & mask_j
            inter_size = bin(inter_mask).count('1')
            
            rem_i_mask = mask_i ^ inter_mask
            rem_j_mask = mask_j ^ inter_mask
            
            forbidden_mask = rem_i_mask | rem_j_mask
            available_mask = universe & ~(forbidden_mask | inter_mask)
            
            need = k - inter_size
            
            if need < 0:
                continue
                
            # Get available indices
            # Optimization: precompute bits for all masks? No, available_mask varies.
            # But we can iterate through bits of available_mask.
            avail_indices = []
            m = available_mask
            while m:
                lsb = m & -m
                idx_bit = (lsb.bit_length() - 1)
                avail_indices.append(idx_bit)
                m ^= lsb
                
            if len(avail_indices) < need:
                continue
                
            # Generate combinations
            for comb in combinations(avail_indices, need):
                mask_c = inter_mask
                for idx_bit in comb:
                    mask_c |= (1 << idx_bit)
                
                if mask_c in mask_to_idx:
                    idx_c = mask_to_idx[mask_c]
                    if idx_c != i and idx_c != j:
                        conf[i][j].append(idx_c)
                        conf[j][i].append(idx_c)

    # Greedy with random restarts
    best_family = []
    best_size = 0
    indices = list(range(num_sets))
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        chosen = set()
        block_count = [0] * num_sets
        is_blocked = [False] * num_sets
        
        for idx in indices:
            if idx in chosen or is_blocked[idx]:
                continue
            
            # Add idx
            chosen.add(idx)
            
            # Update blocks
            # Iterate over previously chosen
            # Note: chosen now contains idx. We need to pair idx with all OTHER chosen.
            # So we can iterate over chosen and skip idx?
            # Or keep a list of chosen indices for faster iteration?
            # Set iteration is fine.
            
            for y in chosen:
                if y == idx:
                    continue
                # Get conflicts for pair (idx, y)
                # conf[idx][y] is the list
                for k in conf[idx][y]:
                    block_count[k] += 1
                    if block_count[k] == 1:
                        is_blocked[k] = True
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
            
    result = []
    for idx in best_family:
        result.append(mask_to_bits[all_sets[idx]])
        
    return result
