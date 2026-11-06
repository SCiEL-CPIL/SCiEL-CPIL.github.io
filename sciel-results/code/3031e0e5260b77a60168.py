"""Phylogenetic topology recovery using NumPy-accelerated Neighbor Joining."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_newick(tree, labels_map):
    """
    Recursively build Newick string from the internal tree structure.
    tree: dict or string node name
    labels_map: maps internal node ids to their structure
    """
    if isinstance(tree, str):
        return tree
    # If it's a tuple (left, right)
    return f"({_build_newick(tree[0], labels_map)},{_build_newick(tree[1], labels_map)})"

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 250.0  # Safety margin before 270s kill
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})();"

    # Convert sequences to integer matrix for fast comparison
    # A:0, C:1, G:2, T:3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    # Efficient conversion using vectorized operations
    # Map characters to integers directly
    # We can use np.frombuffer or just a simple loop since N is small (<=32)
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        # Vectorized conversion
        seq_mat[i] = np.vectorize(base_to_int.get)(seq)
    
    # Compute pairwise Hamming distances
    # (N, L) vs (N, L) -> (N, N, L) differences -> sum over L
    # For N=32, L=2000, this is 32*32*2000 = 2,048,000 elements.
    # int8: ~2MB. This is very fast.
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    # Normalize to p
    p = hamming.astype(np.float64) / L
    
    # Jukes-Cantor Correction
    # d = -0.75 * ln(1 - 4/3 * p)
    # Handle p >= 0.75
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    # Ensure symmetry and zero diagonal
    dist_mat = (dist_mat + dist_mat.T) * 0.5
    np.fill_diagonal(dist_mat, 0.0)
    
    # Initial nodes are the labels
    nodes = labels[:]
    tree_struct = {label: label for label in labels}
    
    # D is the distance matrix indexed by nodes
    # We will maintain a list `nodes` and a matrix `D` where D[i,j] is dist(nodes[i], nodes[j])
    D = dist_mat.copy()
    
    # Map node names to indices for quick lookup?
    # Since we rebuild the list each time, we can just use indices in the current list.
    
    while len(nodes) > 3:
        if time.time() - start_time > time_limit:
            # Fallback: return a star tree or current partial tree?
            # Better to return a star tree for remaining nodes to be valid
            return f"({','.join(nodes)});"

        n = len(nodes)
        
        # Compute row sums r
        r = D.sum(axis=1)
        
        # Calculate Q matrix: Q_ij = (n-2)D_ij - r_i - r_j
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q pair (i, j)
        # Since Q is symmetric, we can just find the global minimum off-diagonal.
        # Using argmin on the flattened array is efficient.
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        
        # Ensure i < j for consistent naming? Not strictly necessary but good for determinism.
        if i > j:
            i, j = j, i
            
        node_i = nodes[i]
        node_j = nodes[j]
        
        # Create new node
        new_node = f"n{len(nodes)}_{i}_{j}"
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        # Determine which indices to keep
        # We want to remove i and j and add new_node at the end.
        # The order of kept nodes matters for the new D matrix.
        
        # Indices to keep from the original list
        kept_indices = [k for k in range(n) if k != i and k != j]
        
        # New nodes list: kept nodes in original order, then new_node
        new_nodes_list = [nodes[k] for k in kept_indices] + [new_node]
        
        # Construct new distance matrix D_new
        # Size: (n-1) x (n-1)
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        # 1. Fill D_new for kept-kept distances
        # Extract submatrix of D for kept_indices
        if kept_indices:
            D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
            D_new[:-1, :-1] = D_keep_keep
        
        # 2. Fill D_new for kept-new_node distances
        # Formula: D_new(k, m) = 0.5 * (D(k, i) + D(k, j) - D(i, j))
        # where k is in kept_indices, m is the new node.
        if kept_indices:
            dist_i_kept = D[kept_indices, i]
            dist_j_kept = D[kept_indices, j]
            dist_ij = D[i, j]
            
            new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
            
            D_new[:-1, -1] = new_branches
            D_new[-1, :-1] = new_branches
        
        # Update state
        nodes = new_nodes_list
        D = D_new
        
    # Final 3 nodes left: a, b, c
    # Connect them in a tree. The topology is (a,(b,c)) or any permutation.
    # Since it's unrooted, the split structure is the same.
    # Newick: (a, (b, c))
    if len(nodes) == 3:
        a, b, c = nodes
        tree_struct['final_root'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final_root'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final_root'] = (tree_struct[a], tree_struct[b])
        final_node = 'final_root'
    else: # 1 node
        final_node = nodes[0]
    
    newick = _build_newick(tree_struct[final_node], tree_struct)
    return newick + ";"

# EVOLVE-BLOCK-END
