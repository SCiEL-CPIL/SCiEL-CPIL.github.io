"""Kissing number in dimension 11: E11 Root System Construction."""
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    D = 11

    # 1. Generate S1: 5 disjoint vectors from 2e_i +/- 2e_j
    # We pick pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    # Signs are + + for simplicity, as any sign works for compatibility with S2
    # and mutual compatibility (disjoint supports).
    S1_vectors = []
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for i, j in pairs:
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        v[j] = 2
        S1_vectors.append(v)
    
    S1_vectors = np.array(S1_vectors)

    # 2. Generate S2: All (+/- 1 ... +/- 1) with even number of -1s
    # There are 2^10 = 1024 such vectors.
    # We can generate by iterating over 10 bits, setting the 11th to ensure even parity.
    # Or just generate all 2^11 and filter.
    S2_list = []
    # Use itertools.product for 11 signs
    # Filter: sum(signs) is even? 
    # Number of -1s is even.
    # If we map 1->1, -1->-1, sum is 11 - 2*k.
    # k is number of -1s.
    # k is even => 11 - 2k is odd?
    # 11 - 0 = 11 (odd)
    # 11 - 2 = 9 (odd)
    # So all such vectors have odd sum.
    # We just need to count the number of -1s.
    
    for signs in itertools.product([1, -1], repeat=D):
        # Count -1s
        if signs.count(-1) % 2 == 0:
            S2_list.append(signs)
            
    S2_vectors = np.array(S2_list, dtype=np.int64)
    n_S2 = len(S2_vectors)
    
    # 3. Build conflict graph for S2
    # Conflict if Hamming distance < 3.
    # Since min dist between even parity vectors is 2 (dist 1 changes parity),
    # Conflict if Hamming distance == 2.
    # Hamming distance 2 <=> Dot product = 11 - 2*2 = 7.
    
    # Compute dot products
    dots = S2_vectors @ S2_vectors.T
    conflict = (dots == 7).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # 4. Greedy + Local Search on S2 conflict graph
    def greedy_select(conflict, n_nodes, rng):
        selected = []
        remaining = np.ones(n_nodes, dtype=bool)
        
        while remaining.any():
            # Degree in remaining subgraph
            deg = np.sum(conflict[:, remaining], axis=1)
            deg[~remaining] = n_nodes + 1
            
            min_deg = deg.min()
            if min_deg >= n_nodes:
                break
                
            candidates_mask = (deg == min_deg) & remaining
            candidates_indices = np.where(candidates_mask)[0]
            
            if len(candidates_indices) == 0:
                break
                
            chosen = candidates_indices[rng.randint(0, len(candidates_indices))]
            
            selected.append(chosen)
            remaining[chosen] = False
            neighbors = np.where(conflict[chosen] > 0)[0]
            remaining[neighbors] = False
            
        return selected

    def local_search(selected_indices, conflict, n_nodes, rng, max_iters=500):
        selected_set = set(selected_indices)
        size = len(selected_set)
        best_size = size
        best_set = list(selected_set)
        
        # Conflict counts for non-selected nodes
        # conflict_counts[j] = number of selected nodes conflicting with j
        conflict_counts = np.zeros(n_nodes, dtype=np.int32)
        for s in selected_set:
            conflict_counts += conflict[s]
            
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try to add
            # Node j is addable if conflict_counts[j] == 0 and j not in selected
            in_selected = np.zeros(n_nodes, dtype=bool)
            for s in selected_set:
                in_selected[s] = True
                
            addable = np.where((conflict_counts == 0) & (~in_selected))[0]
            if len(addable) > 0:
                j = addable[rng.randint(0, len(addable))]
                selected_set.add(j)
                conflict_counts += conflict[j]
                size += 1
                improved = True
                if size > best_size:
                    best_size = size
                    best_set = list(selected_set)
                continue
            
            # Try to swap
            # Pick random i from selected
            i = list(selected_set)[rng.randint(0, len(selected_set)-1)]
            
            old_counts = conflict_counts.copy()
            conflict_counts -= conflict[i]
            selected_set.discard(i)
            
            # Recalculate in_selected
            in_selected = np.zeros(n_nodes, dtype=bool)
            for s in selected_set:
                in_selected[s] = True
                
            addable = np.where((conflict_counts == 0) & (~in_selected))[0]
            
            if len(addable) > 0:
                j = addable[rng.randint(0, len(addable))]
                selected_set.add(j)
                conflict_counts += conflict[j]
                # Size same, but structure changed
                improved = True
            else:
                conflict_counts = old_counts
                selected_set.add(i)
                
        return best_set

    rng = np.random.RandomState(42)
    
    current_best = []
    
    # Run restarts
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
            
        init_set = greedy_select(conflict, n_S2, rng)
        improved_set = local_search(init_set, conflict, n_S2, rng, max_iters=100)
        
        if len(improved_set) > len(current_best):
            current_best = improved_set
            # Optional: print for debugging
            # print(f"Restart {restart}: S2 size {len(current_best)}")

    # 5. Combine S1 and best S2 subset
    if not current_best:
        # Fallback: just S1
        return S1_vectors
        
    S2_selected = S2_vectors[current_best]
    
    # Concatenate
    result = np.vstack([S1_vectors, S2_selected])
    
    return result
