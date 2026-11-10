"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
    # Encode sequences as numpy arrays of integers
    # Map A=0, C=1, G=2, T=3
    # We can use a lookup table for speed if needed, but direct comparison is fine.
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Convert string to bytes for fast numpy conversion
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr
    
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            # Count differences
            diffs = np.count_nonzero(enc[i] != enc[j])
            p = diffs / seq_len
            D[i, j] = D[j, i] = _jc_distance(p)
    return D


def _bionj(D, labels):
    """
    Perform BIONJ (Bootstrap-based Neighbor Joining) on a distance matrix.
    Uses variance-corrected criterion for more accurate topology.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)
    
    while len(node_names) > 3:
        m = len(node_names)
        
        # 1. Row sums
        r = D.sum(axis=1)
        
        # 2. Sum of squares of each row
        S2 = (D ** 2).sum(axis=1)
        
        # 3. Gram matrix (dot products of rows)
        G = D @ D.T
        
        # 4. Compute v_i (variance of distances from i)
        # v_i = (S2_i - r_i^2 / (m-1)) / (m-2)
        v = (S2 - (r ** 2) / (m - 1)) / (m - 2)
        
        # 5. Compute v_ij for all pairs
        # m_diff = r_i - r_j
        m_diff = r[:, None] - r[None, :]
        
        # S2_diff_ij = sum_k (D_ik - D_jk)^2 = S2_i + S2_j - 2 * G_ij
        S2_diff = S2[:, None] + S2[None, :] - 2 * G
        
        # v_ij = (S2_diff - 2*D_ij^2 + m_diff^2/(m-3)) / (m-3)
        # Note: D_ij is the current distance matrix D
        D_sq = D ** 2
        v_ij = (S2_diff - 2 * D_sq + (m_diff ** 2) / (m - 3)) / (m - 3)
        
        # 6. Compute BIONJ criterion Q_ij
        # Q_ij = (m-2)D_ij - r_i - r_j - ((m-2)/(m-3)) * (v_i + v_j - 2*v_ij)
        correction = ((m - 2) / (m - 3)) * (v[:, None] + v[None, :] - 2 * v_ij)
        Q = (m - 2) * D - r[:, None] - r[None, :] - correction
        
        # Set diagonal to inf to avoid self-joining
        np.fill_diagonal(Q, np.inf)
        
        # Find min
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        
        # Create new node
        new_name = f"({node_names[i]},{node_names[j]})"
        
        # Compute new distances
        # d_new_k = 0.5 * (D_i_k + D_j_k - D_i_j)
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        
        # Remove rows/cols i and j, add new row/col
        keep = [k for k in range(m) if k != i and k != j]
        new_m = m - 1
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        
        # Copy existing distances
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        
        # Add new node at position new_m (last)
        new_node_idx = m - 2
        for a, ia in enumerate(keep):
            new_D[a, new_node_idx] = new_dist[ia]
            new_D[new_node_idx, a] = new_dist[ia]
        new_D[new_node_idx, new_node_idx] = 0.0
        
        # Update node_names
        new_names = [node_names[k] for k in keep] + [new_name]
        
        D = new_D
        node_names = new_names
        
    # Now we have 3 nodes
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _bionj_with_deadline(sequences, labels, deadline):
    """Run BIONJ with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    D = _compute_distance_matrix(sequences, labels)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    return _bionj(D, labels)


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Try BIONJ
    result = _bionj_with_deadline(sequences, labels, deadline - start_time)
    if result is not None:
        return result + ";"

    # Fallback: star tree (should not happen given time budget)
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
