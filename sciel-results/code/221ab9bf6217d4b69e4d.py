"""FastUPGMA + NJ hybrid for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _jc_distances(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    # Convert to integer arrays for fast comparison
    # A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = []
    for s in seqs:
        arr = np.array([char_map[c] for c in s], dtype=np.uint8)
        arrs.append(arr)
    
    # Stack for pairwise comparison
    stacked = np.stack(arrs, axis=0)  # shape (n, L)
    
    # Compute pairwise differences using broadcasting
    # diffs[i, j] = number of positions where stacked[i] != stacked[j]
    diffs = np.sum(stacked[:, None, :] != stacked[None, :, :], axis=2)
    
    # Convert to probability
    p = diffs.astype(np.float64) / L
    
    # Apply Jukes-Cantor correction
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        
    # Replace invalid values (inf, nan) with a large number
    d = np.where(np.isfinite(d), d, 3.0)
    
    # Ensure diagonal is 0
    np.fill_diagonal(d, 0.0)
    
    return d


def _nj_step(dist, labels):
    """
    Perform one Neighbor-Joining step.
    Returns: (a_idx, b_idx, new_dist, new_labels)
    where a_idx, b_idx are indices in labels to join,
    new_dist is the updated distance matrix (n-1 x n-1),
    new_labels is the list of remaining labels.
    """
    n = len(labels)
    if n <= 3:
        return None, None, None, None
    
    # Compute Q matrix
    r = np.sum(dist, axis=1)  # row sums
    Q = (n - 2) * dist - np.outer(r, r)
    
    # Find minimum Q entry (off-diagonal)
    min_val = float('inf')
    min_i, min_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] < min_val:
                min_val = Q[i, j]
                min_i, min_j = i, j
    
    a, b = min_i, min_j
    
    # Create new label
    new_label = f"n{len(labels)}_{labels[a]}_{labels[b]}"
    
    # Compute distances from new node to all others
    new_dist = np.zeros((n - 1, n - 1), dtype=np.float64)
    rest = [k for k in range(n) if k not in (a, b)]
    
    # Map old indices to new indices
    new_idx_map = {old: new for new, old in enumerate(rest)}
    
    for k in rest:
        d_k = 0.5 * (dist[a, k] + dist[b, k] - dist[a, b])
        for l in rest:
            if l == k:
                new_dist[new_idx_map[k], new_idx_map[l]] = d_k
            else:
                new_dist[new_idx_map[k], new_idx_map[l]] = dist[k, l]
    
    new_labels = [labels[k] for k in rest] + [new_label]
    # Actually, let's keep the order consistent
    new_labels = [labels[k] for k in rest] + [new_label]
    
    return a, b, new_dist, new_labels


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology using Neighbor-Joining.
    Returns Newick string.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Compute initial distance matrix
    dist = _jc_distances(sequences, labels)
    
    # Track tree topology
    # Each node is either a leaf (original label) or an internal node
    # We'll build the tree by tracking merges
    tree_nodes = {lab: lab for lab in labels}
    
    # Work with indices into a dynamic list of nodes
    node_labels = list(labels)
    current_dist = dist
    
    while len(node_labels) > 3:
        a, b, new_dist, new_labels = _nj_step(current_dist, node_labels)
        if a is None:
            break
        
        # Create the merged subtree
        merged_tree = f"({tree_nodes[node_labels[a]]},{tree_nodes[node_labels[b]]})"
        new_label = new_labels[-1]  # The new label is appended at the end
        tree_nodes[new_label] = merged_tree
        
        # Update node_labels and current_dist
        node_labels = new_labels
        current_dist = new_dist
    
    # Build final tree from remaining 3 nodes
    remaining = [tree_nodes[lab] for lab in node_labels]
    result = "(" + ",".join(remaining) + ");"
    
    return result
# EVOLVE-BLOCK-END
