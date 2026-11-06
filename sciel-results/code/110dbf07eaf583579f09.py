# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 260.0  # wall-clock safety margin
_start_wall = time.time()

def _encode_sequences(sequences, labels):
    """
    Convert dictionary of sequences to a NumPy array of shape (n, L) with values 0-3.
    """
    n = len(labels)
    L = len(sequences[labels[0]])
    mat = np.empty((n, L), dtype=np.uint8)
    
    # Create a translation table for faster mapping
    # A->0, C->1, G->2, T->3
    # Using numpy's frombuffer on bytes might be faster if we map bytes to ints
    # But let's use a simple vectorized approach with searchsorted or a lookup array
    lookup = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        # Convert string to array of byte values
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        # Map using lookup table
        mat[i] = lookup[seq_bytes]
    return mat

def _calc_distances(mat):
    """
    Calculate pairwise JC-corrected distances.
    """
    n, L = mat.shape
    # Calculate mismatches
    # diff[i, j, k] = 1 if mat[i, k] != mat[j, k]
    # We can do this in chunks to avoid huge memory usage if L is large, 
    # but L=2000, n=32 -> 32*32*2000 = 2M booleans = 2MB. Very safe.
    
    # Vectorized mismatch count
    # mat[:, None, :] != mat[None, :, :] creates (n, n, L)
    diff = (mat[:, None, :] != mat[None, :, :])
    mismatches = diff.sum(axis=2)
    
    # Proportion of differences
    p = mismatches.astype(np.float64) / float(L)
    
    # JC Correction: D = -3/4 * ln(1 - 4/3 * p)
    # Handle p=1 (max diff) to avoid log(0)
    p_clipped = np.clip(p, 0.0, 0.75 - 1e-9)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_clipped)
    
    # Ensure symmetry and zero diagonal
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    
    return D

def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining.
    Returns a Newick string.
    """
    n = D.shape[0]
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # We maintain a list of active nodes.
    # Each node is either a leaf (int index) or a subtree (string).
    # We also maintain the current distance matrix for active nodes.
    
    active = list(range(n))
    dist = D.copy()
    # Map from original index to its current representation (string or int)
    # Actually, we can just build the Newick string recursively or iteratively.
    # Let's store the "name" of each current node.
    names = [f"t{label}" for label in labels]
    
    # To handle the distance matrix updates correctly, we need to track which original indices
    # correspond to the current rows/cols in `dist`.
    # Let's use a list `indices` that holds the original indices corresponding to the current dist rows/cols.
    indices = list(range(n))
    
    # We will store the Newick subtrees for each active node.
    # Initially, they are just the leaf names.
    # When we merge i and j, we create a new node with a new name (subtree string).
    
    # To make it simpler, let's just track the topology via the merges.
    # We'll maintain a list of active nodes. Each node has:
    # - id: unique identifier
    # - name: Newick string for this node
    # - dist: distance to other active nodes
    
    # Let's use a dictionary or parallel lists.
    # active_ids: list of current node ids
    # node_names: dict id -> newick string
    # dist_matrix: (m, m) matrix of distances between active_ids
    
    active_ids = list(range(n))
    node_names = {i: labels[i] for i in range(n)}
    dist_mat = D.copy()
    
    next_id = n
    
    while len(active_ids) > 3:
        if time.time() > _DEADLINE - 5:
            break
            
        m = len(active_ids)
        # Calculate Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        # where r[i] = sum(D[i, :])
        
        r = dist_mat.sum(axis=1)
        Q = (m - 2) * dist_mat - r[:, None] - r[None, :]
        
        # Find min Q (i, j) with i != j
        # Set diagonal to inf to ignore
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        # Ensure i != j (should be guaranteed by inf diagonal)
        
        # Get the original indices corresponding to these active positions
        # Note: dist_mat is indexed by position in active_ids, not the id itself.
        # So i and j are positions in the active_ids list.
        
        idx_i = active_ids[i]
        idx_j = active_ids[j]
        
        name_i = node_names[idx_i]
        name_j = node_names[idx_j]
        
        # Calculate branch lengths for the new internal node
        # d(i, j)
        d_ij = dist_mat[i, j]
        
        # r_i = sum of dist_mat[i, :]
        # r_j = sum of dist_mat[j, :]
        # r is already calculated above.
        
        # delta_i = r_i / (m - 2)
        # delta_j = r_j / (m - 2)
        # d(i, new) = 0.5 * d_ij + delta_i
        # d(j, new) = 0.5 * d_ij + delta_j
        
        delta_i = r[i] / (m - 2)
        delta_j = r[j] / (m - 2)
        
        d_i_new = 0.5 * d_ij + delta_i
        d_j_new = 0.5 * d_ij + delta_j
        
        # Clamp to 0
        if d_i_new < 0: d_i_new = 0.0
        if d_j_new < 0: d_j_new = 0.0
        
        # Create new node name
        # Format: (name_i:len_i, name_j:len_j)
        new_name = f"({name_i}:{d_i_new:.4f},{name_j}:{d_j_new:.4f})"
        
        # Update distances to other nodes
        # For any k not i, j:
        # d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        
        # We need to update dist_mat.
        # The new node will replace one of the slots (e.g., i), and j will be removed.
        
        # Calculate new distances for row i (which becomes the new node)
        # k ranges over all active except i, j
        others = [k for k in range(m) if k != i and k != j]
        
        if others:
            # dist_mat[i, others] = 0.5 * (dist_mat[i, others] + dist_mat[j, others] - d_ij)
            dist_mat[i, others] = 0.5 * (dist_mat[i, others] + dist_mat[j, others] - d_ij)
            # Symmetry
            dist_mat[others, i] = dist_mat[i, others]
        
        # Remove j from active list and matrix
        # We need to remove row j and col j from dist_mat
        # And remove j from active_ids
        
        # Create new mask to keep rows/cols
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[j] = False
        
        # Update dist_mat
        dist_mat = dist_mat[np.ix_(keep_mask, keep_mask)]
        
        # Update active_ids
        # active_ids is a list. Remove the element at index j.
        # But wait, if i > j, removing j shifts i.
        # It's safer to reconstruct the list or use indices carefully.
        
        # Let's reconstruct active_ids and node_names mapping?
        # No, node_names is keyed by the *original* id (idx_i, idx_j).
        # The new node has a new id: next_id.
        
        # The position i in the new dist_mat corresponds to the new node.
        # The position of other nodes shifts if j < i.
        
        # Let's update active_ids.
        # The element at position i in active_ids is idx_i.
        # The element at position j in active_ids is idx_j.
        
        # We want the new active_ids to contain:
        # - The new node id at the position where i was (if i < j) or adjusted.
        # - All other nodes except j.
        
        # Let's just build a new list.
        new_active_ids = []
        for k in range(m):
            if k == j:
                continue
            if k == i:
                new_active_ids.append(next_id)
            else:
                new_active_ids.append(active_ids[k])
                
        active_ids = new_active_ids
        node_names[next_id] = new_name
        next_id += 1
        
    # Final step: 3 nodes left
    if len(active_ids) == 3:
        # Get the names
        names_final = [node_names[id] for id in active_ids]
        # Combine them into a star tree (trifurcation) or a binary tree?
        # NJ produces a binary tree. The last 3 nodes should be connected.
        # Usually, the last merge connects two, then the third is attached to the internal node?
        # No, NJ stops at 3 nodes. The topology is determined by the distances.
        # For 3 leaves, the tree is a star. But since we have been building binary trees,
        # the last 3 nodes represent subtrees.
        # We just need to output a Newick string that includes all 3.
        # A valid Newick for 3 leaves is ((A,B),C) or (A,(B,C)).
        # The topology among the last 3 is determined by which two were closest?
        # Actually, the NJ algorithm effectively merges the closest pair at each step.
        # When 3 nodes remain, say A, B, C.
        # The tree is determined by the merges that led to this state.
        # However, the standard NJ output for the last 3 is often just a star or a binary tree based on distances.
        # To be safe and valid, we can compute the distances among the 3 remaining nodes
        # and pick the pair with the smallest distance to be the first merge.
        
        # Get distances among the 3 active nodes
        # active_ids has 3 elements.
        # dist_mat is 3x3.
        
        # Find the pair with min distance
        d01 = dist_mat[0, 1]
        d02 = dist_mat[0, 2]
        d12 = dist_mat[1, 2]
        
        if d01 <= d02 and d01 <= d12:
            # Merge 0 and 1
            n0 = names_final[0]
            n1 = names_final[1]
            n2 = names_final[2]
            # Calculate branch lengths
            d_01 = d01
            # delta_0 = (d01 + d02 - d12) / 2
            # delta_1 = (d01 + d12 - d02) / 2
            # d_0_new = 0.5 * d01 + delta_0 = (2*d01 + d01 + d02 - d12) / 4 ? No.
            # Standard NJ branch length formula for the last step:
            # len_0 = 0.5 * (d01 + d02 - d12)
            # len_1 = 0.5 * (d01 + d12 - d02)
            # len_2 = 0.5 * (d02 + d12 - d01)
            
            l0 = 0.5 * (d01 + d02 - d12)
            l1 = 0.5 * (d01 + d12 - d02)
            l2 = 0.5 * (d02 + d12 - d01)
            
            if l0 < 0: l0 = 0.0
            if l1 < 0: l1 = 0.0
            if l2 < 0: l2 = 0.0
            
            return f"(({n0}:{l0:.4f},{n1}:{l1:.4f}),{n2}:{l2:.4f});"
            
        elif d02 <= d01 and d02 <= d12:
            # Merge 0 and 2
            n0 = names_final[0]
            n1 = names_final[1]
            n2 = names_final[2]
            
            d_02 = d02
            l0 = 0.5 * (d02 + d01 - d12)
            l2 = 0.5 * (d02 + d12 - d01)
            l1 = 0.5 * (d01 + d12 - d02)
            
            if l0 < 0: l0 = 0.0
            if l2 < 0: l2 = 0.0
            if l1 < 0: l1 = 0.0
            
            return f"(({n0}:{l0:.4f},{n2}:{l2:.4f}),{n1}:{l1:.4f});"
            
        else:
            # Merge 1 and 2
            n0 = names_final[0]
            n1 = names_final[1]
            n2 = names_final[2]
            
            d_12 = d12
            l1 = 0.5 * (d12 + d01 - d02)
            l2 = 0.5 * (d12 + d02 - d01)
            l0 = 0.5 * (d01 + d02 - d12)
            
            if l1 < 0: l1 = 0.0
            if l2 < 0: l2 = 0.0
            if l0 < 0: l0 = 0.0
            
            return f"(({n1}:{l1:.4f},{n2}:{l2:.4f}),{n0}:{l0:.4f});"
            
    # Fallback (should not happen)
    return "(" + ",".join(node_names[id] for id in active_ids) + ");"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
        
    # Encode sequences
    mat = _encode_sequences(sequences, labels)
    L = mat.shape[1]
    
    if L == 0:
        # No data, return star tree
        return "(" + ",".join(labels) + ");"
        
    # Calculate distances
    D = _calc_distances(mat)
    
    # Perform NJ
    newick = _neighbor_joining(D, labels)
    
    return newick

# EVOLVE-BLOCK-END
