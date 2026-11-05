import numpy as np
import time
import math

def _jc_distance_matrix(sequences, labels):
    """
    Calculate the Jukes-Cantor corrected distance matrix using NumPy.
    """
    N = len(labels)
    L = len(sequences[0])
    
    seq_array = np.array(sequences, dtype='U1')
    mismatches = (seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :]).sum(axis=2)
    p = mismatches / float(L)
    p_safe = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0/3.0) * p_safe)
    D[p >= 0.75] = 3.0
    return D

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    N = len(labels)
    
    if N == 1:
        return f"{labels[0]};\n"
    if N == 2:
        return f"({labels[0]},{labels[1]});\n"
    
    D = _jc_distance_matrix([sequences[l] for l in labels], labels)
    
    # Nodes are indices 0..N-1
    active = list(range(N))
    dist = D.copy()
    tree_strs = [l for l in labels]
    
    while len(active) > 3:
        if time.time() > deadline:
            break
            
        n = len(active)
        
        # Extract submatrix for active nodes
        # Note: active is a list of indices into the *current* dist matrix?
        # No, in the previous implementation, I was mixing original indices and current matrix indices.
        # It is cleaner to maintain `dist` as a matrix corresponding exactly to the `active` list order.
        # When we join, we update `dist` to the new size.
        
        # So `dist` is always (len(active), len(active)).
        
        totals = dist.sum(axis=1)
        
        # Find min Q
        min_q = float('inf')
        best_i, best_j = -1, -1
        
        # Vectorized Q calculation?
        # Q[i,j] = (n-2)*dist[i,j] - totals[i] - totals[j]
        # We can compute this for all i,j and find the min.
        # Q_matrix = (n - 2) * dist - totals[:, np.newaxis] - totals[np.newaxis, :]
        # Set diagonal to inf to avoid self-pairing.
        # Then find argmin.
        
        Q = (n - 2) * dist - totals[:, np.newaxis] - totals[np.newaxis, :]
        np.fill_diagonal(Q, float('inf'))
        
        # Find index of min value
        min_val = np.min(Q)
        # Get indices
        # Since Q is symmetric, we can just find the first min.
        # But we need i < j to avoid duplicates? 
        # If we take the global min, it might be (j,i) where j>i.
        # It doesn't matter which order we pick for a and b, as long as we remove both.
        
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        best_i, best_j = idx[0], idx[1]
        
        a, b = best_i, best_j
        da_b = dist[a, b]
        
        # Create new node string
        new_node_str = f"({tree_strs[a]},{tree_strs[b]})"
        
        # Calculate distances from new node to all other active nodes
        # Exclude a and b
        mask = [k for k in range(n) if k != a and k != b]
        
        # d(new, k) = 0.5 * (d(a, k) + d(b, k) - d(a, b))
        # dist[a, mask] gives distances from a to all k in mask
        # dist[b, mask] gives distances from b to all k in mask
        new_dists = 0.5 * (dist[a, mask] + dist[b, mask] - da_b)
        
        # Update active list
        new_active = [k for k in range(n) if k != a and k != b]
        new_active.append(-1) # Placeholder for new node index in the *new* matrix
        
        # Update tree_strs
        new_tree_strs = [tree_strs[k] for k in range(n) if k != a and k != b]
        new_tree_strs.append(new_node_str)
        
        # Update distance matrix
        # New size (n-1, n-1)
        # Copy existing distances for mask
        # mask is list of indices into old dist
        # We need to map old indices to new indices.
        # The new matrix will have rows/cols corresponding to `mask` followed by the new node.
        
        # Extract submatrix
        # np.ix_ with mask
        new_dist_core = dist[np.ix_(mask, mask)]
        
        new_dist = np.zeros((n - 1, n - 1))
        new_dist[:n-2, :n-2] = new_dist_core
        
        # Fill new node distances
        new_idx = n - 2
        new_dist[new_idx, :n-2] = new_dists
        new_dist[:n-2, new_idx] = new_dists
        
        active = new_active
        tree_strs = new_tree_strs
        dist = new_dist
        
    # Handle remaining 3 nodes
    if len(active) == 3:
        i, j, k = active[0], active[1], active[2]
        final_str = f"({tree_strs[i]},{tree_strs[j]},{tree_strs[k]})"
    elif len(active) == 2:
        final_str = f"({tree_strs[active[0]]},{tree_strs[active[1]]})"
    elif len(active) == 1:
        final_str = f"{tree_strs[active[0]]}"
    else:
        # Fallback
        final_str = f"({','.join(tree_strs)})"
        
    return final_str + ";"
