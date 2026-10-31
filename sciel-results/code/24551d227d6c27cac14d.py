import numpy as np
import itertools
import time
import sys
import scipy.sparse as sp

D = 11

def generate_candidates():
    """Generate all vectors of weight 3 and 4 in {-1, 0, 1}^D"""
    candidates = []
    # Weight 3
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    # Weight 4
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts_csr, deadline, rng):
    """
    Greedy Maximum Independent Set using CSR conflict matrix.
    """
    N = conflicts_csr.shape[0]
    alive = np.ones(N, dtype=bool)
    # degrees are the number of conflicts with alive nodes
    # We can compute initial degrees from CSR
    degrees = np.array(conflicts_csr.sum(axis=1)).flatten().astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        # We need to update degrees.
        # Instead of updating all, we can just pick min from current alive.
        # But degrees are stale if we don't update.
        # Let's update degrees properly.
        
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        # Get neighbors of chosen from CSR
        start = conflicts_csr.indptr[chosen]
        end = conflicts_csr.indptr[chosen + 1]
        neighbors = conflicts_csr.indices[start:end]
        
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        # For each node in to_remove, subtract 1 from degree of its neighbors
        # This is O(sum of degrees of removed nodes)
        for node in to_remove:
            s = conflicts_csr.indptr[node]
            e = conflicts_csr.indptr[node + 1]
            nbrs = conflicts_csr.indices[s:e]
            # Only update if alive
            mask = alive[nbrs]
            if np.any(mask):
                degrees[nbrs[mask]] -= 1

    return selected_indices

def local_improve(selected, vectors, conflicts_csr, deadline, rng):
    """
    Try to improve the selected set by swapping out one vector for another.
    """
    N = conflicts_csr.shape[0]
    selected_set = set(selected)
    
    # Precompute unselected
    unselected = np.array([i for i in range(N) if i not in selected_set])
    if len(unselected) == 0:
        return selected
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(selected)):
            if time.time() > deadline:
                return selected
            v_out = selected[idx]
            other_selected = [s for s in selected if s != v_out]
            
            if len(other_selected) == 0:
                compatible_unselected = unselected
            else:
                # Check conflicts between unselected and other_selected
                # For each u in unselected, check if it conflicts with any s in other_selected
                # We can do this by iterating u and checking neighbors
                # Or vectorize:
                # conflicts_csr[unselected, other_selected] is not directly supported in CSR for submatrix easily without converting.
                # Let's use a loop for unselected, but break early.
                compatible = []
                for u in unselected:
                    # Get neighbors of u
                    s_idx = conflicts_csr.indptr[u]
                    e_idx = conflicts_csr.indptr[u + 1]
                    nbrs = conflicts_csr.indices[s_idx:e_idx]
                    # Check if any neighbor is in other_selected
                    # Convert other_selected to set for O(1) lookup? 
                    # other_selected is small (K ~ 1000).
                    # We can check intersection.
                    # np.intersect1d(nbrs, other_selected)
                    if len(np.intersect1d(nbrs, other_selected, assume_unique=True)) == 0:
                        compatible.append(u)
                
                if len(compatible) == 0:
                    continue
                
                compatible_unselected = np.array(compatible)
            
            if len(compatible_unselected) > 0:
                u_in = compatible_unselected[rng.integers(len(compatible_unselected))]
                # Swap
                selected[idx] = u_in
                selected_set.discard(v_out)
                selected_set.add(u_in)
                # Update unselected
                unselected = np.array([i for i in range(N) if i not in selected_set])
                improved = True
                break
    
    return selected

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # Generate candidates (Weight 3 and 4)
    vectors = generate_candidates()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Determine norms squared
    norms_sq = np.sum(vectors * vectors, axis=1)
    
    # Conflict logic:
    # dist^2 >= max_norm_sq
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # ||u||^2 + ||v||^2 - 2 u.v >= max_norm_sq
    # 2 u.v <= ||u||^2 + ||v||^2 - max_norm_sq
    
    max_norm_sq = np.max(norms_sq) # Should be 4
    
    # Threshold for dot product:
    # u.v <= (||u||^2 + ||v||^2 - max_norm_sq) / 2
    
    # We can compute the allowed dot product for each pair (i, j)
    # allowed[i, j] = (norms_sq[i] + norms_sq[j] - max_norm_sq) / 2
    
    norm_sum = norms_sq[:, None] + norms_sq[None, :]
    allowed_dots = (norm_sum - max_norm_sq) / 2.0
    
    # Conflict if dots > allowed_dots
    # Note: dots are integers, allowed_dots might be float (e.g., 1.5)
    # If allowed is 1.5, dot must be <= 1.
    # If allowed is 2.0, dot must be <= 2.
    
    conflicts = (dots > allowed_dots).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    # Convert to CSR
    conflicts_csr = sp.csr_matrix(conflicts)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS restarts
    n_restarts = 0
    while time.time() < deadline - 10:
        current_set = greedy_mis(conflicts_csr, deadline - 10, rng)
        n_restarts += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set
        
        # If we found a large set, do local improvement
        if len(best_set) > 0 and time.time() < deadline - 5:
            improved_set = local_improve(list(best_set), vectors, conflicts_csr, deadline - 2, rng)
            if len(improved_set) > len(best_set):
                best_set = improved_set
    
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
