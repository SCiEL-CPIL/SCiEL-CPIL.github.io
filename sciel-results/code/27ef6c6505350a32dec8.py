"""Phylogenetic topology recovery using optimized Neighbor Joining with vectorized distance computation."""
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
    if len(tree) == 2:
        left = _build_newick(tree[0], labels_map)
        right = _build_newick(tree[1], labels_map)
        return f"({left},{right})"
    return str(tree)

def _compute_total_tree_length(tree_struct, root, dist_mat, label_to_idx):
    """Compute the total tree length using the distance matrix."""
    def get_subtree_taxa(node):
        if isinstance(node, str):
            return {node}
        return get_subtree_taxa(node[0]) | get_subtree_taxa(node[1])
    
    total_length = 0.0
    
    def process_node(node):
        nonlocal total_length
        if isinstance(node, str):
            return
        left_taxa = get_subtree_taxa(node[0])
        right_taxa = get_subtree_taxa(node[1])
        left_size = len(left_taxa)
        right_size = len(right_taxa)
        
        if left_size > 0 and right_size > 0:
            sum_dist = 0.0
            for i_str in left_taxa:
                i_idx = label_to_idx[i_str]
                for j_str in right_taxa:
                    j_idx = label_to_idx[j_str]
                    sum_dist += dist_mat[i_idx, j_idx]
            
            split_dist = sum_dist / (left_size * right_size)
            total_length += split_dist
        
        process_node(node[0])
        process_node(node[1])
    
    process_node(tree_struct[root])
    return total_length

def _run_nj(dist_mat, labels, start_time, time_limit):
    """Run Neighbor Joining algorithm. Returns (tree_struct, final_node) or (None, None) on timeout."""
    n_labels = len(labels)
    nodes = labels[:]
    D = dist_mat.copy()
    tree_struct = {label: label for label in labels}
    
    while len(nodes) > 3:
        if time.time() - start_time > time_limit:
            return None, None
        
        n = len(nodes)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        if i == j:
            continue
        
        node_i = nodes[i]
        node_j = nodes[j]
        
        new_node = f"n{len(nodes)}_{i}_{j}"
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        kept_indices = np.where(keep_mask)[0]
        
        new_nodes_list = [nodes[k] for k in kept_indices]
        new_nodes_list.append(new_node)
        
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
        D_new[:-1, :-1] = D_keep_keep
        
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        nodes = new_nodes_list
        D = D_new
    
    if len(nodes) == 3:
        a, b, c = nodes
        tree_struct['final_root'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final_root'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final_root'] = (tree_struct[a], tree_struct[b])
        final_node = 'final_root'
    else:
        final_node = nodes[0]
    
    return tree_struct, final_node

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 260.0  # 270s budget, leave 10s safety margin
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})();"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    seq_chars = np.array([list(sequences[label]) for label in labels], dtype='U1')
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for base, val in base_to_int.items():
        seq_mat[seq_chars == base] = val
    
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    p = hamming.astype(np.float64) / L
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_jc = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    dist_jc = 0.5 * (dist_jc + dist_jc.T)
    np.fill_diagonal(dist_jc, 0.0)
    
    dist_ham = hamming.astype(np.float64)
    dist_ham = 0.5 * (dist_ham + dist_ham.T)
    np.fill_diagonal(dist_ham, 0.0)
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Run NJ on JC-corrected distances
    tree_jc, final_jc = _run_nj(dist_jc, labels, start_time, time_limit)
    
    # Check time before running second NJ
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        # Run NJ on Hamming distances
        tree_ham, final_ham = _run_nj(dist_ham, labels, start_time, time_limit)
    else:
        tree_ham, final_ham = None, None
    
    # Handle timeout cases
    if tree_jc is None or final_jc is None:
        if tree_ham is not None and final_ham is not None:
            newick = _build_newick(tree_ham[final_ham], tree_ham)
            return newick + ";"
        return f"({','.join(labels)});"
    
    if tree_ham is None or final_ham is None:
        newick = _build_newick(tree_jc[final_jc], tree_jc)
        return newick + ";"
    
    # Score both trees using JC-corrected distances
    score_jc = _compute_total_tree_length(tree_jc, final_jc, dist_jc, label_to_idx)
    score_ham = _compute_total_tree_length(tree_ham, final_ham, dist_jc, label_to_idx)
    
    if score_jc <= score_ham:
        newick = _build_newick(tree_jc[final_jc], tree_jc)
    else:
        newick = _build_newick(tree_ham[final_ham], tree_ham)
    
    return newick + ";"

# EVOLVE-BLOCK-END
