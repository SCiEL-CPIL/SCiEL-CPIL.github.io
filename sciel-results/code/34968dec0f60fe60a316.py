import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate weight-2 vectors in {-1, 0, 1}^11 (Norm^2 = 2)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)
            
    # Generate weight-1 vectors in {-1, 0, 1}^11 (Norm^2 = 1)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    N2 = len(vectors_v2)
    N1 = len(vectors_v1)
    
    if N4 == 0:
        return np.vstack([vectors_v2, vectors_v1])
    
    # Compute conflict matrices
    # V4-V4: conflict if dot > 2
    dots_44 = vectors_v4 @ vectors_v4.T
    conflicts_44 = dots_44 > 2
    np.fill_diagonal(conflicts_44, False)
    
    # V2-V2: conflict if dot > 1
    dots_22 = vectors_v2 @ vectors_v2.T
    conflicts_22 = dots_22 > 1
    np.fill_diagonal(conflicts_22, False)
    
    # V1-V1: conflict if dot > 0 (i.e., dot >= 1, same direction)
    dots_11 = vectors_v1 @ vectors_v1.T
    conflicts_11 = dots_11 > 0
    np.fill_diagonal(conflicts_11, False)
    
    # V4-V2: conflict if dot > 1
    dots_42 = vectors_v4 @ vectors_v2.T
    conflicts_42 = dots_42 > 1
    
    # V4-V1: conflict if dot > 0 (i.e., dot >= 1)
    dots_41 = vectors_v4 @ vectors_v1.T
    conflicts_41 = dots_41 > 0
    
    # V2-V1: conflict if dot > 0 (i.e., dot >= 1)
    dots_21 = vectors_v2 @ vectors_v1.T
    conflicts_21 = dots_21 > 0
    
    # Use greedy minimum-degree for V4 vectors with multiple restarts
    rng = np.random.default_rng(42)
    best_selected4 = np.zeros(N4, dtype=bool)
    best_count4 = 0
    
    for restart in range(10):
        if time.time() > deadline - 30:
            break
        
        rng_r = np.random.default_rng(42 + restart * 7)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        degrees = conflicts_44.sum(axis=1).astype(np.int64)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng_r.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts_44[chosen] | (np.arange(N4) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts_44[remaining][:, to_remove].sum(axis=1)
        
        count = selected.sum()
        if count > best_count4:
            best_count4 = count
            best_selected4 = selected.copy()
    
    # Local search: try 1-swap and 2-swap improvements
    for iteration in range(20):
        if time.time() > deadline - 20:
            break
        
        improved = False
        
        # Try 1-swap: add one non-selected vertex
        non_selected = np.where(~best_selected4)[0]
        for v in non_selected:
            sel_neighbors = np.where(best_selected4 & conflicts_44[v])[0]
            net_gain = 1 - len(sel_neighbors)
            if net_gain > 0:
                best_selected4[v] = True
                for sn in sel_neighbors:
                    best_selected4[sn] = False
                best_count4 += net_gain
                improved = True
                break
        
        if not improved:
            # Try 2-swap: add two non-selected vertices
            non_selected = np.where(~best_selected4)[0]
            if len(non_selected) >= 2:
                found = False
                for _ in range(200):
                    if time.time() > deadline - 20:
                        break
                    i, j = rng.choice(non_selected, size=2, replace=False)
                    if conflicts_44[i, j]:
                        continue
                    sel_neigh_i = np.where(best_selected4 & conflicts_44[i])[0]
                    sel_neigh_j = np.where(best_selected4 & conflicts_44[j])[0]
                    combined = np.unique(np.concatenate([sel_neigh_i, sel_neigh_j]))
                    net_gain = 2 - len(combined)
                    if net_gain > 0:
                        best_selected4[i] = True
                        best_selected4[j] = True
                        for sn in combined:
                            best_selected4[sn] = False
                        best_count4 += net_gain
                        improved = True
                        found = True
                        break
                if found:
                    continue
        
        if not improved:
            break
    
    result_v4 = vectors_v4[best_selected4]
    
    # Select V2 vectors compatible with selected V4
    selected_v4_indices = np.where(best_selected4)[0]
    if len(selected_v4_indices) > 0:
        v2_conflicts_with_v4 = conflicts_42[selected_v4_indices, :].any(axis=0)
    else:
        v2_conflicts_with_v4 = np.zeros(N2, dtype=bool)
    
    v2_available = ~v2_conflicts_with_v4
    alive2 = v2_available.copy()
    selected2 = np.zeros(N2, dtype=bool)
    degrees2 = conflicts_22.sum(axis=1).astype(np.int64)
    
    while np.any(alive2):
        alive_indices2 = np.where(alive2)[0]
        if len(alive_indices2) == 0:
            break
        alive_degrees2 = degrees2[alive_indices2]
        min_deg2 = np.min(alive_degrees2)
        cands2 = alive_indices2[alive_degrees2 == min_deg2]
        chosen2 = cands2[rng.integers(len(cands2))]
        
        selected2[chosen2] = True
        to_remove2 = np.where(conflicts_22[chosen2] | (np.arange(N2) == chosen2))[0]
        alive2[to_remove2] = False
        remaining2 = np.where(alive2)[0]
        if len(remaining2) > 0:
            degrees2[remaining2] -= conflicts_22[remaining2][:, to_remove2].sum(axis=1)
    
    result_v2 = vectors_v2[selected2]
    
    # Select V1 vectors compatible with selected V4 and V2
    if len(selected_v4_indices) > 0:
        v1_conflicts_with_v4 = conflicts_41[selected_v4_indices, :].any(axis=0)
    else:
        v1_conflicts_with_v4 = np.zeros(N1, dtype=bool)
    
    selected_v2_indices = np.where(selected2)[0]
    if len(selected_v2_indices) > 0:
        v1_conflicts_with_v2 = conflicts_21[selected_v2_indices, :].any(axis=0)
    else:
        v1_conflicts_with_v2 = np.zeros(N1, dtype=bool)
    
    v1_available = ~(v1_conflicts_with_v4 | v1_conflicts_with_v2)
    alive1 = v1_available.copy()
    selected1 = np.zeros(N1, dtype=bool)
    degrees1 = conflicts_11.sum(axis=1).astype(np.int64)
    
    while np.any(alive1):
        alive_indices1 = np.where(alive1)[0]
        if len(alive_indices1) == 0:
            break
        alive_degrees1 = degrees1[alive_indices1]
        min_deg1 = np.min(alive_degrees1)
        cands1 = alive_indices1[alive_degrees1 == min_deg1]
        chosen1 = cands1[rng.integers(len(cands1))]
        
        selected1[chosen1] = True
        to_remove1 = np.where(conflicts_11[chosen1] | (np.arange(N1) == chosen1))[0]
        alive1[to_remove1] = False
        remaining1 = np.where(alive1)[0]
        if len(remaining1) > 0:
            degrees1[remaining1] -= conflicts_11[remaining1][:, to_remove1].sum(axis=1)
    
    result_v1 = vectors_v1[selected1]
    
    # Combine all results
    parts = [result_v4]
    if len(result_v2) > 0:
        parts.append(result_v2)
    if len(result_v1) > 0:
        parts.append(result_v1)
    
    final_vectors = np.vstack(parts)
    
    # Ensure at least one vector
    if len(final_vectors) == 0:
        return np.array([[1] + [0]*10], dtype=np.int64)
        
    # Repair step: Remove conflicting pairs
    # Calculate norms squared
    norms_sq = np.sum(final_vectors**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    if len(final_vectors) > 1:
        # Compute pairwise squared distances
        # This can be memory intensive for very large k, but k is likely < 1000
        diff = final_vectors[:, None, :] - final_vectors[None, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Identify conflicts: distance squared < max norm squared
        conflicts = dists_sq < max_norm_sq
        np.fill_diagonal(conflicts, False)
        
        # Greedy removal of conflicts
        # We want to keep as many as possible.
        # Simple approach: iterate through conflicts, remove one vector.
        # To be faster, we can use a heap or just pick min degree in conflict graph.
        # Given the time limit, a simple loop is safer.
        
        valid = np.ones(len(final_vectors), dtype=bool)
        
        # Build adjacency list or just use the matrix
        # We will iteratively find a conflict, remove one, and update.
        # To avoid O(N^3), we can just scan the matrix once and mark removals.
        # However, removing a node might resolve multiple conflicts.
        
        # Heuristic: Remove nodes that are involved in many conflicts first?
        # Or just remove any node involved in a conflict.
        
        # Efficient check:
        # While there is any True in conflicts:
        #   Find first True (i, j)
        #   Mark j as invalid
        #   Set conflicts[i, :] = False, conflicts[:, i] = False (optional, as we check valid mask)
        #   Set conflicts[j, :] = False, conflicts[:, j] = False
        #   valid[j] = False
        
        # We need to be careful not to pick an already invalid node.
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Find any active conflict
            # We can use np.where on the intersection of conflicts and valid masks
            # But conflicts is a matrix.
            # Let's just iterate through indices.
            # For efficiency, we can maintain a list of active indices.
            # But simple loop over np.where might be fine for small k.
            
            # To speed up, we can check if any conflict exists among valid nodes.
            # active_conflicts = conflicts[valid][:, valid]
            # This is a copy.
            
            # Let's just pick the first conflict in the original matrix
            # and check if both are valid.
            
            # Optimization: Pre-compute list of conflicts?
            # No, just loop.
            
            # Find first conflict
            # np.where(conflicts) is O(N^2) memory/time.
            # We can just loop i, j.
            
            # Since N is small, we can do:
            found = False
            for i in range(len(final_vectors)):
                if not valid[i]:
                    continue
                for j in range(i + 1, len(final_vectors)):
                    if not valid[j]:
                        continue
                    if conflicts[i, j]:
                        # Conflict found
                        # Remove j (arbitrary choice, or remove one with higher degree in conflict graph)
                        # For simplicity, remove j
                        valid[j] = False
                        changed = True
                        found = True
                        break
                if found:
                    break
            
            if not found:
                break
                
        final_vectors = final_vectors[valid]
        
    if len(final_vectors) == 0:
        return np.array([[1] + [0]*10], dtype=np.int64)
        
    return final_vectors
