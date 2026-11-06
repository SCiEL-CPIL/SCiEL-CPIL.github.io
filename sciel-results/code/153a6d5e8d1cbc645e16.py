"""Optimized Neighbor-Joining with vectorized distance computation and O(n^3) NJ loop."""
import numpy as np


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using binary matrix and vectorized NumPy."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Encode each base as 2-bit binary: A=00, C=01, G=10, T=11
    # This allows us to use XOR to detect differences
    # Create two binary matrices: one for bit 0, one for bit 1
    bit0 = np.zeros((n, length), dtype=np.int8)
    bit1 = np.zeros((n, length), dtype=np.int8)
    
    for i, seq in enumerate(seq_list):
        for j, ch in enumerate(seq):
            if ch == 'A':
                pass  # 00
            elif ch == 'C':
                bit0[i, j] = 1  # 01
            elif ch == 'G':
                bit1[i, j] = 1  # 10
            elif ch == 'T':
                bit0[i, j] = 1
                bit1[i, j] = 1  # 11
    
    # Two sequences differ at position k if bit0 differs OR bit1 differs
    # diff_count = sum over k of (bit0[i,k] != bit0[j,k]) + (bit1[i,k] != bit1[j,k])
    # But we need to be careful: if both bits differ, that's still 1 difference, not 2
    # Actually, for Jukes-Cantor we just need the count of positions where bases differ
    # XOR approach: base differs if (bit0 XOR bit0) OR (bit1 XOR bit1)
    # Let's just compute it directly with broadcasting for simplicity and correctness
    
    # Re-encode as single integer per base for cleaner comparison
    base_vals = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        for j, ch in enumerate(seq):
            if ch == 'A':
                base_vals[i, j] = 0
            elif ch == 'C':
                base_vals[i, j] = 1
            elif ch == 'G':
                base_vals[i, j] = 2
            elif ch == 'T':
                base_vals[i, j] = 3
    
    # Vectorized pairwise difference count
    # (n, n, length) -> sum over last axis
    diff_matrix = (base_vals[:, None, :] != base_vals[None, :, :]).sum(axis=2).astype(np.float64)
    
    p = diff_matrix / length
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Neighbor-Joining algorithm with O(n^3) complexity using in-place distance matrix updates.
    Returns a Newick string.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # We'll use a fixed-size (2n x 2n) distance matrix to avoid reallocation
    # Nodes 0..n-1 are leaves, nodes n..2n-2 are internal nodes
    max_nodes = 2 * n - 1
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = d
    
    # Newick string for each node
    tree = [f"t{i}" for i in range(n)]
    tree.extend([''] * (max_nodes - n))
    
    # Active nodes: initially 0..n-1
    active = list(range(n))
    num_active = n
    
    # Track which original leaf each node corresponds to (for final assembly)
    # Actually, we just need the Newick strings
    
    while num_active > 2:
        # Compute row sums for active nodes
        # r[i] = sum of D[i, j] for all active j
        r = np.zeros(max_nodes, dtype=np.float64)
        for i in active:
            r[i] = D[i, active].sum()
        
        # Find pair (a, b) minimizing Q(a,b) = (num_active - 2) * D[a,b] - r[a] - r[b]
        best_q = float('inf')
        best_a = -1
        best_b = -1
        
        for i_idx in range(num_active):
            a = active[i_idx]
            for j_idx in range(i_idx + 1, num_active):
                b = active[j_idx]
                q = (num_active - 2) * D[a, b] - r[a] - r[b]
                if q < best_q:
                    best_q = q
                    best_a = a
                    best_b = b
        
        a = best_a
        b = best_b
        
        # Compute branch lengths
        delta = (r[a] - r[b]) / (2.0 * (num_active - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        # Create new internal node
        new_node = max_nodes - 1  # Use the last available slot, but we need to track used nodes
        # Actually, let's use a counter
        # We'll track next_free_node
        # Let me restructure: use a node_id counter
        
        # For now, let's just append to tree and track active properly
        # Let me rewrite this more carefully
        
        # New node index: we'll use n + (n - num_active) as the internal node index
        # Since we've already created (n - num_active) internal nodes
        new_node = n + (n - num_active)
        
        # Update distance matrix: D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        for x in active:
            if x != a and x != b:
                D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
                D[x, new_node] = D[new_node, x]
        
        # Build Newick string for new node
        tree[new_node] = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Update active list: remove a and b, add new_node
        active = [x for x in active if x != a and x != b]
        active.append(new_node)
        num_active -= 1
    
    # Combine last two nodes
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
