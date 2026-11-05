"""Distance-based phylogenetic tree reconstruction using Neighbor-Joining with Jukes-Cantor correction."""
import numpy as np
import math


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0..t(n-1)) to DNA strings over ACGT
        
    Returns:
        Newick string representing the unrooted tree topology
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Convert sequences to numpy arrays for faster distance computation
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    # Pre-convert sequences to integer arrays for faster comparison
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_seqs = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            int_seqs[i, j] = base_to_int[c]
    
    # Compute pairwise Jukes-Cantor corrected distances using vectorized operations
    # For each pair (i,j), count mismatches across all sites
    dist = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff_count = np.sum(int_seqs[i] != int_seqs[j])
            p = diff_count / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    
    # Neighbor-Joining algorithm using numpy arrays for efficiency
    # We maintain:
    #   active: list of node ids currently in the NJ process
    #   D: 2D numpy array of distances between all nodes (including internal)
    #   tree_str: dict mapping node_id -> Newick string for that subtree
    #   node_to_idx: mapping from node id to its index in the D array
    
    total_nodes = 2 * n - 1  # maximum number of nodes (n leaves + n-1 internal)
    D_full = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    
    # Initialize D_full with the initial distances
    D_full[:n, :n] = dist
    
    tree_str = {i: labels[i] for i in range(n)}
    active = list(range(n))
    next_node_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Extract the submatrix for active nodes
        D_sub = D_full[np.ix_(active, active)]
        
        # Compute row sums (total distance from each node to all others)
        r = D_sub.sum(axis=1)
        
        # Find the pair (a, b) that minimizes the NJ criterion:
        # Q(a,b) = (m-2)*D[a,b] - r[a] - r[b]
        Q = (m - 2) * D_sub - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Set diagonal to inf to avoid self-pairs
        np.fill_diagonal(Q, np.inf)
        
        # Find the minimum Q value (off-diagonal)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        idx_a, idx_b = min_idx
        
        a = active[idx_a]
        b = active[idx_b]
        
        # Create a new internal node joining a and b
        new_node = next_node_id
        next_node_id += 1
        
        # Compute branch lengths
        delta_a = 0.5 * (D_sub[idx_a, idx_b] + (r[idx_a] - r[idx_b]) / (m - 2))
        delta_b = D_sub[idx_a, idx_b] - delta_a
        
        # Ensure non-negative (clamp small negatives to 0)
        if delta_a < 0:
            delta_a = 0.0
        if delta_b < 0:
            delta_b = 0.0
        
        # Build the subtree string
        tree_str[new_node] = f"({tree_str[a]}:{delta_a:.4f},{tree_str[b]}:{delta_b:.4f})"
        
        # Update distances from new_node to all remaining active nodes
        remaining = [x for x in active if x not in (a, b)]
        
        if remaining:
            # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            a_idx_in_active = idx_a
            b_idx_in_active = idx_b
            
            # Get distances from a and b to remaining nodes
            # We need to find the indices of remaining nodes in the active list
            rem_indices = [i for i, x in enumerate(active) if x in remaining]
            
            da_to_rem = D_sub[a_idx_in_active, rem_indices]
            db_to_rem = D_sub[b_idx_in_active, rem_indices]
            dab = D_sub[idx_a, idx_b]
            
            new_dists = 0.5 * (da_to_rem + db_to_rem - dab)
            
            # Set distances in D_full
            for k, rem_node in enumerate(remaining):
                D_full[new_node, rem_node] = new_dists[k]
                D_full[rem_node, new_node] = new_dists[k]
        
        # Remove a and b from active, add new_node
        active = remaining + [new_node]
    
    # Final step: join the remaining 3 nodes into an unrooted tree
    a, b, c = active
    
    dab = D_full[a, b]
    dac = D_full[a, c]
    dbc = D_full[b, c]
    
    # Branch lengths for the 3-taxon unrooted tree
    len_a = max(0.0, (dab + dac - dbc) / 2.0)
    len_b = max(0.0, (dab + dbc - dac) / 2.0)
    len_c = max(0.0, (dac + dbc - dab) / 2.0)
    
    # Build final Newick
    result = f"({tree_str[a]}:{len_a:.4f},{tree_str[b]}:{len_b:.4f},{tree_str[c]}:{len_c:.4f})"
    
    return result
