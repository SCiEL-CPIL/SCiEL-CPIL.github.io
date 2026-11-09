"""Rewritten program with vectorized Neighbor Joining and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def reconstruct(sequences):
    start_time = time.time()
    # Deadline guard: 160s hard limit to ensure we finish well before 180s/270s
    DEADLINE = 160.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # Map labels to indices for easy array access
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Create sequence matrix: (n, L)
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    # Convert to numpy array of characters or use string operations
    # For Jukes-Cantor, we just need pairwise differences.
    # Efficient way: compare each sequence against all others.
    # Since n is small (<=32) and L is moderate (<=2000), we can do this.
    
    # Vectorized distance calculation
    # seq_matrix is shape (n, L)
    seq_matrix = np.array([list(s) for s in seq_list])
    
    # Calculate pairwise Hamming distances
    # diff_matrix[i, j] = number of positions where seq_i != seq_j
    # We can compute this by broadcasting
    # diff_matrix = np.sum(seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :], axis=2)
    # This creates an (n, n, L) intermediate which might be large if L is big, but n is small.
    # n=32, L=2000 -> 32*32*2000 = 2M elements. This is fine.
    
    diff_matrix = np.sum(seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :], axis=2)
    
    # Convert to Jukes-Cantor distances
    p = diff_matrix / float(L)
    
    # JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 case (log domain issue)
    # np.log(1 - 4p/3) will be -inf if 1-4p/3 <= 0.
    # We clip p to a value slightly less than 0.75 to avoid -inf, or handle explicitly.
    # The original code capped at 3.0.
    
    # Create a mask for valid logs
    valid_mask = (1.0 - 4.0 * p / 3.0) > 0
    # For invalid ones, we'll assign a large distance (e.g., 3.0)
    
    D = np.zeros((n, n))
    D[valid_mask] = -0.75 * np.log(1.0 - 4.0 * p[valid_mask] / 3.0)
    D[~valid_mask] = 3.0
    
    # Diagonal should be 0
    np.fill_diagonal(D, 0.0)
    
    # Neighbor Joining Algorithm
    # We maintain a list of active nodes.
    # Initially, all taxa are active.
    # Each active node has a "tree" string representation.
    
    active_nodes = list(range(n))
    # tree_strs[i] is the Newick string for the subtree rooted at node i
    # Initially, just the label
    tree_strs = [f"t{idx}" for idx in range(n)]
    
    # We need to track which original taxa are in which subtree for potential debugging, 
    # but for NJ we just need the distance matrix updates.
    
    current_n = n
    next_node_id = n
    
    while len(active_nodes) > 3:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            # Return the current best tree (which is just the current active nodes connected in a star or whatever structure we have so far)
            # Actually, if we break here, we have a partial tree. 
            # The safest bet is to return a star tree of the remaining active nodes, 
            # or just the last fully constructed tree if we had one.
            # But since we update in place, let's just construct a tree from the remaining active nodes as a star.
            if len(active_nodes) == 1:
                return tree_strs[active_nodes[0]] + ";"
            else:
                # Connect remaining nodes in a star
                inner = ",".join(tree_strs[i] for i in active_nodes)
                return f"({inner});"
        
        m = len(active_nodes)
        
        # Calculate row sums (Q matrix preparation)
        # r_i = sum_j D[i, j]
        # D is the full (n, n) matrix, but we only care about active nodes.
        # Let's extract the submatrix for active nodes for clarity and speed.
        idx = active_nodes
        D_sub = D[np.ix_(idx, idx)]
        
        r = np.sum(D_sub, axis=1)
        
        # Calculate Q matrix: Q[i, j] = (m-2)*D[i, j] - r[i] - r[j]
        # We want to find the pair (i, j) with the minimum Q.
        
        # Vectorized Q calculation
        # Q[i, j] = (m-2)*D_sub[i, j] - r[i] - r[j]
        Q = (m - 2) * D_sub - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Set diagonal to inf so we don't pick (i, i)
        np.fill_diagonal(Q, np.inf)
        
        # Find the index of the minimum value
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        # Map back to original node IDs
        i_node = idx[i_idx]
        j_node = idx[j_idx]
        
        # Create new node
        new_node = next_node_id
        next_node_id += 1
        
        # Calculate branch lengths
        # delta = Q[i, j] / (2 * (m - 2))
        delta = Q[i_idx, j_idx] / (2 * (m - 2))
        
        # d(i, new) = D[i, j] / 2 - delta
        # d(j, new) = D[i, j] / 2 + delta
        d_ij = D_sub[i_idx, j_idx]
        
        d_i_new = d_ij / 2.0 - delta
        d_j_new = d_ij / 2.0 + delta
        
        # Ensure non-negative (numerical stability)
        if d_i_new < 0: d_i_new = 0.0
        if d_j_new < 0: d_j_new = 0.0
        
        # Update tree string
        # The new node connects to the subtrees of i and j
        # We don't strictly need branch lengths in the output string for RF distance, 
        # but the interface allows them. Standard Newick with branch lengths is safe.
        # However, the prompt says "branch lengths optional". 
        # Adding them is fine, but simpler is better for parsing if the evaluator is strict.
        # Let's stick to topology only to be safe, or include them if it doesn't hurt.
        # The current seed program does NOT include branch lengths in the final string.
        # Let's follow the seed program's format: "(A,B)"
        
        tree_strs.append(f"({tree_strs[i_node]},{tree_strs[j_node]})")
        
        # Update distance matrix
        # D[new, k] = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        
        # We need to update the full D matrix to include the new node.
        # The active nodes are idx.
        # The new distances are for the new node against all OTHER active nodes (excluding i and j).
        
        # Let's construct the new row for the new node.
        # It will have length (m - 2) corresponding to the remaining active nodes.
        
        # Identify remaining active nodes
        remaining_idx = [k for k in range(m) if k != i_idx and k != j_idx]
        remaining_nodes = [idx[k] for k in remaining_idx]
        
        # Calculate new distances
        # D_new_k = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        
        # D_sub[i_idx, remaining_idx] gives D[i, k] for k in remaining
        # D_sub[j_idx, remaining_idx] gives D[j, k] for k in remaining
        
        d_i_rem = D_sub[i_idx, remaining_idx]
        d_j_rem = D_sub[j_idx, remaining_idx]
        
        new_dists = 0.5 * (d_i_rem + d_j_rem - d_ij)
        
        # Now we need to update the main D matrix.
        # The main D matrix is (n_total, n_total).
        # We are adding a new row/col at index `new_node`.
        
        # We need to set D[new_node, remaining_nodes] = new_dists
        # and D[remaining_nodes, new_node] = new_dists
        
        # Also, we need to remove i_node and j_node from the "active" set, 
        # but their entries in D remain for potential future reference? 
        # No, in standard NJ, we just work with the active set.
        # The D matrix is fixed size (n, n) where n is initial taxa count + created nodes.
        # We just need to ensure that when we extract D_sub next time, 
        # we only look at active nodes.
        
        # So, we just need to fill in the new row/col for the new node in the global D matrix.
        
        D[new_node, remaining_nodes] = new_dists
        D[remaining_nodes, new_node] = new_dists
        
        # The diagonal for the new node
        D[new_node, new_node] = 0.0
        
        # Update active nodes list
        # Remove i_node and j_node, add new_node
        active_nodes = [x for x in active_nodes if x != i_node and x != j_node]
        active_nodes.append(new_node)
        
    # Construct final tree
    if len(active_nodes) == 3:
        a, b, c = active_nodes
        # Unrooted tree of 3 nodes is just a star
        # (A, (B, C)) or similar. Topology is the same.
        return f"({tree_strs[a]},{tree_strs[b]},{tree_strs[c]});"
    elif len(active_nodes) == 2:
        return f"({tree_strs[active_nodes[0]]},{tree_strs[active_nodes[1]]});"
    elif len(active_nodes) == 1:
        return f"{tree_strs[active_nodes[0]]};"
    else:
        # Fallback
        inner = ",".join(tree_strs[i] for i in active_nodes)
        return f"({inner});"

# EVOLVE-BLOCK-END
