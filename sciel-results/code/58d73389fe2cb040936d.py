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
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    # ASCII: A=65, C=67, G=71, T=84
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr
    
    # Vectorized pairwise difference counting
    # enc shape: (n, L). Compute all pairwise differences at once.
    # diffs[i,j] = number of positions where enc[i] != enc[j]
    # Use broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This can be memory-heavy for large n, so do it in chunks if needed.
    
    # For n up to 32 and L up to 2000: 32*32*2000 = 2M bytes - fine
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        # Count differences between row i and all other rows
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs
    
    # Convert to proportions and apply JC correction
    p = diff_matrix.astype(np.float64) / seq_len
    # Apply JC correction element-wise
    # Clip p to avoid log of non-positive
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    # For p >= 0.75, set to 3.0
    D[p >= 0.75] = 3.0
    # Ensure symmetry and zero diagonal
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels, deadline=None):
    """
    Perform neighbor joining on a distance matrix D (numpy array, symmetric, zero diag).
    labels: list of taxon names corresponding to rows/cols of D.
    deadline: optional wall-clock deadline (time.time() value) to stop early.
    Returns a Newick string (no trailing semicolon).
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
        # Check deadline
        if deadline is not None and time.time() > deadline:
            # Return star tree with remaining nodes
            return "(" + ",".join(node_names) + ")"
        
        m = len(node_names)
        # Compute row sums
        r = D.sum(axis=1)  # shape (m,)

        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        # Set diagonal to inf
        np.fill_diagonal(Q, np.inf)
        # Find min
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        # Create new node
        new_name = f"({node_names[i]},{node_names[j]})"

        # Compute new distances
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        # Build new distance matrix using fancy indexing
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1
        
        # Extract submatrix for kept nodes
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]
        
        # Add new node distances
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        # Update node_names
        new_names = [node_names[k] for k in keep] + [new_name]

        D = new_D
        node_names = new_names

    # Now we have 3 nodes
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _nj_with_deadline(sequences, labels, deadline):
    """Run NJ with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    D = _compute_distance_matrix(sequences, labels)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    # Pass deadline to NJ so it can stop early if needed
    return _neighbor_joining(D, labels, deadline=start + deadline - 2)


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

    # Try neighbor joining
    result = _nj_with_deadline(sequences, labels, deadline - start_time)
    if result is not None:
        # Ensure all taxa appear exactly once
        return result + ";"

    # Fallback: star tree (should not happen given time budget)
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
