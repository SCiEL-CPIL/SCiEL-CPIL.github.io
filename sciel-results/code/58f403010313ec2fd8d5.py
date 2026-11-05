import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Pre-computation of Q3 masks
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Generate local Q3 edge sets on 0..7
    local_q3_edge_sets = set()
    for perm in __import__('itertools').permutations(range(8)):
        edges = []
        for u, v in q3_edges:
            pu = perm[u]
            pv = perm[v]
            if pu > pv: pu, pv = pv, pu
            edges.append((pu, pv))
        local_q3_edge_sets.add(frozenset(edges))
        
    local_q3_list = list(local_q3_edge_sets)
    
    # Map global edges to indices
    edge_idx_map = {}
    edges_list = []
    idx = 0
    for u in range(16):
        for v in range(u + 1, 16):
            edge_idx_map[(u, v)] = idx
            edges_list.append((u, v))
            idx += 1
            
    NUM_EDGES = 120
    subsets_8 = list(combinations(range(16), 8))
    
    # Pre-allocate NumPy arrays for masks
    # We don't know the exact count yet, but it's 12870 * 840 = 10808400
    total_q3s = len(subsets_8) * len(local_q3_list)
    q3_low = np.zeros(total_q3s, dtype=np.uint64)
    q3_high = np.zeros(total_q3s, dtype=np.uint64)
    
    # List of lists for edge indices
    edge_indices = [[] for _ in range(NUM_EDGES)]
    
    mask_count = 0
    for S in subsets_8:
        if time.time() > deadline - 30.0: # Reserve time for search
            # If we run out of time, we might have incomplete precomputation
            # We should handle this, but let's assume it fits.
            pass
        for local_edges in local_q3_list:
            mask_low = 0
            mask_high = 0
            for u, v in local_edges:
                gu = S[u]
                gv = S[v]
                if gu > gv: gu, gv = gv, gu
                e_idx = edge_idx_map[(gu, gv)]
                edge_indices[e_idx].append(mask_count)
                if e_idx < 64:
                    mask_low |= (1 << e_idx)
                else:
                    mask_high |= (1 << (e_idx - 64))
            q3_low[mask_count] = mask_low
            q3_high[mask_count] = mask_high
            mask_count += 1
            
    # Trim arrays if we didn't fill them (due to time break)
    if mask_count < total_q3s:
        q3_low = q3_low[:mask_count]
        q3_high = q3_high[:mask_count]
        for i in range(NUM_EDGES):
            edge_indices[i] = np.array(edge_indices[i], dtype=np.int32)
    else:
        for i in range(NUM_EDGES):
            edge_indices[i] = np.array(edge_indices[i], dtype=np.int32)
            
    if time.time() > deadline - 10.0:
        # Fallback
        return [[a, b] for a in range(3) for b in range(3, 16)]

    # 2. Greedy Construction with NumPy checks
    best_edges = []
    best_count = 0
    
    num_restarts = 50 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(r)
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        current_mask_low = 0
        current_mask_high = 0
        current_edges = []
        
        for e_idx in edge_order:
            e_bit_low = 1 << e_idx if e_idx < 64 else 0
            e_bit_high = 1 << (e_idx - 64) if e_idx >= 64 else 0
            
            # Get candidates
            cand_indices = edge_indices[e_idx]
            
            # If no candidates, it's safe
            if len(cand_indices) == 0:
                current_mask_low |= e_bit_low
                current_mask_high |= e_bit_high
                current_edges.append(e_idx)
                continue
                
            # Vectorized check
            # We need to check if (mask & ~current) == e_bit
            # mask_low & ~current_mask_low == e_bit_low
            # mask_high & ~current_mask_high == e_bit_high
            
            cand_low = q3_low[cand_indices]
            cand_high = q3_high[cand_indices]
            
            # Note: ~current_mask_low in Python/NumPy for uint64 requires care with signs
            # But since we are using uint64, bitwise not works as expected in C-style?
            # Actually, np.uint64 ~ x is (2**64 - 1) - x.
            # We want to check if the bits in cand_low that are NOT in current_mask_low
            # are EXACTLY e_bit_low.
            
            # diff_low = cand_low & (~current_mask_low)
            # But ~current_mask_low flips all 64 bits.
            # We only care about the bits that are in cand_low.
            # So: (cand_low & ~current_mask_low) == e_bit_low
            
            # However, e_bit_low is 0 if e_idx >= 64.
            # If e_idx < 64, e_bit_high is 0.
            
            # Condition:
            # (cand_low & ~current_mask_low) == e_bit_low AND (cand_high & ~current_mask_high) == e_bit_high
            
            # Optimization:
            # If e_idx < 64:
            #   e_bit_high = 0.
            #   We need (cand_high & ~current_mask_high) == 0.
            #   This means all bits in cand_high must be in current_mask_high.
            
            # Let's compute the "missing" bits.
            # missing_low = cand_low & ~current_mask_low
            # missing_high = cand_high & ~current_mask_high
            
            # We want missing_low == e_bit_low and missing_high == e_bit_high.
            
            # Using NumPy:
            # ~current_mask_low is a scalar.
            
            missing_low = cand_low & (~current_mask_low)
            missing_high = cand_high & (~current_mask_high)
            
            # Check equality
            # Note: e_bit_low is a Python int, convert to np.uint64 for comparison?
            # NumPy handles it.
            
            match = (missing_low == e_bit_low) & (missing_high == e_bit_high)
            
            if not np.any(match):
                # Safe to add
                current_mask_low |= e_bit_low
                current_mask_high |= e_bit_high
                current_edges.append(e_idx)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    # Convert best_edges indices back to [u, v] pairs
    result = []
    for e_idx in best_edges:
        u, v = edges_list[e_idx]
        result.append([u, v])
        
    return result
