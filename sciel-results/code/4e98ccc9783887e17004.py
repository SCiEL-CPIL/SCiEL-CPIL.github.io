"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.array([list(map(base_to_int.__getitem__, seq)) for seq in seq_list], dtype=np.int8)
    
    # Compute pairwise Hamming distances via broadcasting
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    p = diff_matrix / length
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Neighbor-Joining algorithm on a distance matrix d of shape (n, n).
    Returns a Newick string.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # D will be updated in-place; we track which original indices are still active
    D = d.copy()
    # tree[i] stores the Newick subtree for node i (i in 0..n-1 are leaves)
    tree = [f"t{i}" for i in range(n)]
    
    # active: list of current node indices (initially 0..n-1)
    active = list(range(n))
    # next_node: counter for new internal nodes
    next_node = n
    
    while len(active) > 2:
        n_active = len(active)
        
        # Compute row sums for active nodes: r[k] = sum over active j of D[active[k], active[j]]
        # Use vectorized sum over the active submatrix
        subD = D[np.ix_(active, active)]
        r = subD.sum(axis=1)
        
        # Q matrix: Q[i,j] = (n_active - 2) * D[i,j] - r[i] - r[j]
        # Only need upper triangle for pair selection
        Q = (n_active - 2) * subD - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum Q in upper triangle
        i_idx, j_idx = np.unravel_index(np.argmin(Q), Q.shape)
        if j_idx < i_idx:
            i_idx, j_idx = j_idx, i_idx
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Compute branch lengths
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (n_active - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        len_a = max(len_a, 0.0)
        len_b = max(len_b, 0.0)
        
        # Create new node
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Remove a and b from active, add new node
        new_active = [x for x in active if x != a and x != b]
        
        # Compute distances from new node to remaining active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        for x in new_active:
            D[a, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, a] = D[a, x]
        
        # Reuse index a for the new node
        tree[a] = new_label
        active = new_active + [a]
    
    # Combine the last two nodes
    i, j = active
    d_ij = D[i, j]
    half = d_ij / 2.0
    newick = f"({tree[i]}:{half:.4f},{tree[j]}:{half:.4f})"
    
    return newick + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings
        
    Returns:
        Newick string
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Run Neighbor-Joining
    newick = _neighbor_joining(D)
    
    return newick
