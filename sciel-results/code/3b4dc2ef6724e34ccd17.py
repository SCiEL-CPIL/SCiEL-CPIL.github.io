import numpy as np
import math
import time

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        if n_taxa == 1:
            return f"({labels[0]});"
        else:
            return f"({labels[0]},{labels[1]});"

    # Map sequences to integer arrays for faster comparison
    # A:0, C:1, G:2, T:3
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_array = np.zeros((n_taxa, len(next(iter(sequences.values()))), ), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j, char in enumerate(s):
            seq_array[i, j] = seq_map[char]

    # Compute pairwise Hamming distances
    # (n_taxa, n_taxa)
    # We can do this by broadcasting
    # diff[i, j, k] = seq_array[i, k] != seq_array[j, k]
    # This is n^2 * L memory, which for n=32, L=2000 is 32*32*2000 = 2M ints, fine.
    # But we can do it in chunks or just use a loop if memory is tight, but 2M is small.
    # Actually, np.not_equal(seq_array[:, None, :], seq_array[None, :, :]).sum(axis=2)
    # This creates a (n, n, L) array.
    
    # More memory efficient:
    # dist_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    # for i in range(n_taxa):
    #    for j in range(i+1, n_taxa):
    #       dist_matrix[i,j] = np.sum(seq_array[i] != seq_array[j])
    
    # Let's use the vectorized way, it's fast enough for n=32
    # Create a 3D array of comparisons
    # To save memory, we can do it in blocks or just accept the memory.
    # 32*32*2000 * 1 byte = 2MB. Very safe.
    
    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    hamming = diff.sum(axis=2)
    
    # Apply Jukes-Cantor correction
    # p = hamming / L
    L = seq_array.shape[1]
    p = hamming / L
    
    # Handle p=0 and p=1
    # JC(d) = -0.75 * ln(1 - 4/3 * p)
    # If p=0, d=0
    # If p=1, d=inf (or large number)
    
    d_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    
    # Mask for valid p (0 < p < 0.75)
    # p can be exactly 0.
    # p > 0.75 is unlikely for real data but possible in simulation if branch lengths are huge.
    # If p >= 0.75, the log argument is <= 0.
    # We cap p at 0.7499 or handle it.
    
    valid_mask = (p > 0) & (p < 0.75)
    
    # Calculate for valid
    d_valid = -0.75 * np.log(1 - (4.0 / 3.0) * p[valid_mask])
    d_matrix[valid_mask] = d_valid
    
    # For p=0, d=0 (already initialized)
    # For p >= 0.75, set to a large value or max observed + epsilon
    # If p == 1, distance is infinite. In NJ, this might cause issues.
    # We'll set it to a large constant.
    large_val = 10.0 # Arbitrary large value, or max(d_valid)*1.5 if any valid
    if np.any(valid_mask):
        large_val = np.max(d_valid) * 1.5
    else:
        large_val = 1.0
        
    invalid_mask = p >= 0.75
    d_matrix[invalid_mask] = large_val
    
    # Ensure diagonal is 0
    np.fill_diagonal(d_matrix, 0.0)
    
    # Neighbor Joining
    # active nodes
    nodes = list(range(n_taxa))
    # trees dict: node_id -> newick_string
    trees = {i: labels[i] for i in range(n_taxa)}
    
    # We need to maintain the distance matrix for the active nodes.
    # As we join nodes, the size of the matrix shrinks.
    # It's easier to keep the full matrix and just track which indices are active.
    # Or, copy the submatrix. Copying is O(N^2), which is fine.
    
    # Let's use a list of active indices and a dictionary of current distances between them?
    # No, keeping a numpy array and updating it is faster.
    
    # We will maintain a list `active` of current node IDs (original indices or new IDs).
    # And a dictionary `dist` for distances between active nodes?
    # Or just use the original indices and a mapping.
    
    # Let's use a standard approach:
    # active: list of node ids
    # D: numpy array of shape (len(active), len(active))
    
    active = list(range(n_taxa))
    D = d_matrix.copy()
    
    next_id = n_taxa
    
    while len(active) > 3:
        if time.time() > deadline:
            # Fallback: return a star tree or the current partial tree?
            # Better to return a valid tree.
            # If we break here, we have a partial tree.
            # We should finish the NJ quickly or return a star tree.
            # For safety, return a star tree connecting all remaining active nodes.
            # But we need to map back to original labels.
            # This is complex. Let's just ensure the loop finishes.
            # With N=32, the loop runs 29 times. It should be very fast.
            pass
            
        n = len(active)
        
        # Row sums
        r = D.sum(axis=1)
        
        # Q matrix
        # Q[i, j] = (n - 2) * D[i, j] - r[i] - r[j]
        Q = (n - 2) * D - r[:, None] - r[None, :]
        
        # Find min Q
        # We want the pair (i, j) with min Q[i, j] where i != j
        # Mask diagonal
        Q_masked = Q.copy()
        np.fill_diagonal(Q_masked, np.inf)
        
        min_val = np.min(Q_masked)
        min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
        i, j = min_idx
        
        # Get the original node ids
        node_i = active[i]
        node_j = active[j]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Update trees
        trees[new_id] = f"({trees[node_i]},{trees[node_j]})"
        
        # Remove i and j from active, add new_id
        # We need to update the distance matrix D.
        # The new row/col for new_id should be calculated for all other active nodes.
        
        # Indices of remaining nodes
        remaining_indices = [k for k in range(n) if k not in (i, j)]
        
        # Calculate new distances
        # D[new, k] = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        
        # We will construct a new D matrix of size (n-1, n-1)
        # The new active list will be: remaining nodes + new_id
        
        # Let's build the new D
        new_n = n - 1
        new_D = np.zeros((new_n, new_n), dtype=np.float64)
        
        # Map old indices to new indices
        # The new active list order: remaining_indices (in original order) + [new_id]
        # But we need to map the distances correctly.
        
        # Let's create a list of new active nodes
        new_active = [active[k] for k in remaining_indices] + [new_id]
        
        # Fill new_D for the remaining-remaining part
        # For each pair in remaining_indices, copy from D
        for a_idx, a in enumerate(remaining_indices):
            for b_idx, b in enumerate(remaining_indices):
                new_D[a_idx, b_idx] = D[a, b]
        
        # Fill new_D for the new node (last row/col)
        # The new node is at index new_n - 1
        for a_idx, a in enumerate(remaining_indices):
            dist_new_a = 0.5 * (D[i, a] + D[j, a] - D[i, j])
            new_D[a_idx, new_n - 1] = dist_new_a
            new_D[new_n - 1, a_idx] = dist_new_a
            
        new_D[new_n - 1, new_n - 1] = 0.0
        
        # Update state
        active = new_active
        D = new_D
        
    # Final step: 3 nodes remain
    # They form a star tree (unrooted)
    # Connect them in a triangle? No, unrooted tree with 3 leaves is a star.
    # The topology is ((a,b),c) or similar.
    # In Newick, we just join them.
    # The order doesn't matter for topology score if it's a star.
    # But we need to return a valid Newick string.
    
    # The remaining 3 nodes are in `active`
    # We just need to combine their trees
    # e.g. (tree1, tree2, tree3) is not valid Newick.
    # It must be (tree1, (tree2, tree3)) or similar.
    
    # Let's just pick the first two, join them, then join with the third.
    # (a, b, c) -> ((a, b), c)
    
    # However, the branch lengths are not strictly required, but the topology is.
    # For 3 taxa, the only unrooted topology is the star.
    
    # Construct the final Newick
    # We have trees for each node in active.
    
    # Pick 0, 1, 2
    t0 = trees[active[0]]
    t1 = trees[active[1]]
    t2 = trees[active[2]]
    
    # Join 0 and 1
    join_01 = f"({t0},{t1})"
    
    # Join with 2
    final_tree = f"({join_01},{t2})"
    
    return final_tree + ";"
