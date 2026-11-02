import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build compatibility matrix: compat[i,j] = True if M[i,j] <= 2
    # Note: M[i,i] = 4, so we must exclude self or handle it.
    # We want i != j.
    compat = (M <= 2)
    # Ensure diagonal is False for the "available" mask logic, 
    # but for the final check we just need pairwise.
    # Let's set diagonal to False so we don't select the same vector twice.
    np.fill_diagonal(compat, False)

    # Build conflict graph adjacency: A[i,j] = 1 if i and j are NOT compatible (i != j)
    # We want a large independent set in the conflict graph = large clique in compat graph
    # Equivalently, large independent set in conflict graph
    A = (~compat).astype(np.float64)
    np.fill_diagonal(A, 0)
    
    # Degree matrix
    deg = A.sum(axis=1)
    
    # Graph Laplacian L = D - A
    L = np.diag(deg) - A
    
    best_set = None
    best_count = 0

    # Strategy 1: Spectral relaxation
    # For Max-Cut style problems, we can use eigenvectors of the Laplacian
    # to find a good partition. For independent set, we want to maximize
    # the number of nodes with no edges between them.
    
    # Compute leading eigenvectors of L
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # The smallest eigenvalues correspond to the most "coherent" subsets
        # For finding large independent sets, we can try different eigenvector-based approaches
        
        # Approach: Use the second smallest eigenvector (Fiedler vector)
        # to partition the graph, then take the larger side
        if len(eigenvalues) >= 2:
            fiedler = eigenvectors[:, 1]
            # Partition: nodes with positive fiedler value vs negative
            pos_mask = fiedler > 0
            neg_mask = fiedler < 0
            
            # Try both partitions and greedy on each
            for mask in [pos_mask, neg_mask]:
                if time.time() > deadline - 5:
                    break
                subset_idx = np.where(mask)[0]
                if len(subset_idx) < 2:
                    continue
                
                # Greedy on this subset
                selected = []
                available = np.ones(len(subset_idx), dtype=bool)
                
                for i, idx in enumerate(subset_idx):
                    if not available[i]:
                        continue
                    selected.append(idx)
                    # Update available
                    for j in range(len(subset_idx)):
                        if available[j] and not compat[idx, subset_idx[j]]:
                            available[j] = False
                    
                    if not np.any(available):
                        break
                
                count = len(selected)
                if count > best_count:
                    best_count = count
                    best_set = selected[:]
        
        # Approach 2: Use multiple eigenvectors and combine
        # Try using the first k eigenvectors to create a spectral embedding
        for k in [2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            if len(eigenvectors) < k:
                break
                
            # Take first k eigenvectors (smallest eigenvalues)
            embedding = eigenvectors[:, :k]
            
            # Normalize rows
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding_normed = embedding / norms
            
            # Cluster using simple threshold on pairwise distances
            # Or use the sign of each component to create binary features
            
            # Simple approach: use the sign of the first few components
            for comp_idx in range(min(k, 4)):
                if time.time() > deadline - 5:
                    break
                sign_mask = embedding[:, comp_idx] > 0
                subset_idx = np.where(sign_mask)[0]
                if len(subset_idx) < 2:
                    continue
                
                # Greedy on this subset
                selected = []
                available = np.ones(len(subset_idx), dtype=bool)
                
                for i, idx in enumerate(subset_idx):
                    if not available[i]:
                        continue
                    selected.append(idx)
                    for j in range(len(subset_idx)):
                        if available[j] and not compat[idx, subset_idx[j]]:
                            available[j] = False
                    
                    if not np.any(available):
                        break
                
                count = len(selected)
                if count > best_count:
                    best_count = count
                    best_set = selected[:]
    except Exception:
        pass

    # Strategy 2: Local search improvement (swap-based)
    # Start from the best set found, try to improve by swapping one node with two others
    
    if best_set is not None:
        selected = best_set[:]
        
        # Multiple local search iterations
        for iteration in range(100):
            if time.time() > deadline - 10:
                break
                
            improved = False
            selected_set = set(selected)
            
            # Try adding a node that's compatible with all selected
            for idx in range(N):
                if idx in selected_set:
                    continue
                # Check if idx is compatible with all selected
                compatible_with_all = True
                for s in selected:
                    if not compat[idx, s]:
                        compatible_with_all = False
                        break
                
                if compatible_with_all:
                    selected.append(idx)
                    selected_set.add(idx)
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one node, add two others
                for remove_idx in range(len(selected)):
                    if time.time() > deadline - 10:
                        break
                    
                    removed = selected[remove_idx]
                    temp_selected = selected[:remove_idx] + selected[remove_idx+1:]
                    temp_set = set(temp_selected)
                    
                    # Find nodes compatible with all temp_selected but not with removed
                    candidates_to_add = []
                    for idx in range(N):
                        if idx in temp_set or idx == removed:
                            continue
                        compatible_with_all_temp = True
                        for s in temp_selected:
                            if not compat[idx, s]:
                                compatible_with_all_temp = False
                                break
                        if compatible_with_all_temp:
                            candidates_to_add.append(idx)
                    
                    # Try to add two mutually compatible candidates
                    for i in range(len(candidates_to_add)):
                        for j in range(i+1, len(candidates_to_add)):
                            a = candidates_to_add[i]
                            b = candidates_to_add[j]
                            if compat[a, b]:
                                # Check if this improves the set
                                new_selected = temp_selected + [a, b]
                                if len(new_selected) > len(selected):
                                    selected = new_selected
                                    improved = True
                                    break
                        if improved:
                            break
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        best_count = len(selected)
        best_set = selected[:]
    else:
        # Fallback: greedy
        rng = np.random.default_rng(42)
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            if trial == 0:
                order = np.arange(N)
            else:
                order = rng.permutation(N)

            selected = []
            available = np.ones(N, dtype=bool)
            
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                available = available & compat[idx, :]
                if not np.any(available):
                    break

            count = len(selected)
            if count > best_count:
                best_count = count
                best_set = selected[:]

    # If we found a set, return it
    if best_set is not None:
        return candidates[best_set].astype(np.int64)

    # Fallback: just the Type 1 vectors (22 vectors)
    # They are all orthogonal, so dot product 0 <= 2.
    return candidates[:22].astype(np.int64)
