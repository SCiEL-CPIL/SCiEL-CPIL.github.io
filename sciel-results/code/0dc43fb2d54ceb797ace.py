"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Encode sequences as integer arrays: A=0, C=1, G=2, T=3
    base_to_int = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)
    base_map = np.zeros(256, dtype=np.int8)
    base_map[base_to_int] = [0, 1, 2, 3]
    
    # Create matrix of shape (n, length) using vectorized lookup
    seq_matrix = np.empty((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        seq_matrix[i] = base_map[np.frombuffer(seq.encode(), dtype=np.uint8)]
    
    # Compute pairwise Hamming distances using matrix multiplication trick
    # For each pair (i,j), count mismatches
    # diff_matrix[i,j] = sum_k (seq_matrix[i,k] != seq_matrix[j,k])
    # Use: (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2)
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Compute p = diff / length
    p = diff_matrix / length
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (saturation) by capping
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Neighbor-Joining algorithm on a distance matrix d of shape (n, n).
    Returns a Newick string. Uses vectorized operations for speed.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # Work with a copy
    D = d.copy()
    
    # Track the current tree structure for each node
    tree = [f"t{i}" for i in range(n)]
    
    # Active nodes
    active = list(range(n))
    
    while len(active) > 2:
        n_active = len(active)
        
        # Build sub-matrix of active nodes
        D_sub = D[np.ix_(active, active)]
        
        # Compute row sums r[i] = sum_j D_sub[i, j] for i in active
        r = D_sub.sum(axis=1)
        
        # Find the pair (a, b) that minimizes:
        # Q(a, b) = (n_active - 2) * D[a, b] - r[a] - r[b]
        # Vectorized: Q[i,j] = (n_active-2)*D_sub[i,j] - r[i] - r[j]
        Q = (n_active - 2) * D_sub - r[:, None] - r[None, :]
        
        # Set diagonal to inf to exclude self-pairs
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum Q value (upper triangle only)
        # Use argmin on the full matrix, then map back
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        best_i, best_j = min_idx
        
        a = active[best_i]
        b = active[best_j]
        
        # Compute branch lengths
        delta = (r[best_i] - r[best_j]) / (2.0 * (n_active - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        # Ensure non-negative
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)
        
        # Create new node
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Create new active list without a and b, add new node
        new_active = [x for x in active if x not in (a, b)]
        new_node = max(active) + 1  # Use a new index
        
        # Compute distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        # Vectorized computation
        if new_active:
            new_D_row = 0.5 * (D[a, new_active] + D[b, new_active] - D[a, b])
        
        # Extend D to include new node
        old_n = D.shape[0]
        new_D = np.zeros((old_n + 1, old_n + 1))
        new_D[:old_n, :old_n] = D
        
        if new_active:
            new_D[new_node, new_active] = new_D_row
            new_D[new_active, new_node] = new_D_row
        
        D = new_D
        tree.append(new_label)
        active = new_active + [new_node]
    
    # Combine the last two nodes
    if len(active) == 2:
        i, j = active
        d_ij = D[i, j]
        half = d_ij / 2.0
        newick = f"({tree[i]}:{half:.4f},{tree[j]}:{half:.4f})"
    else:
        newick = tree[active[0]]
    
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
