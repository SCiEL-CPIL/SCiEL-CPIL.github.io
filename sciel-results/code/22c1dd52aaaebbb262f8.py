"""Reconstruct phylogenetic topology from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 255.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _neighbor_joining(D, labels):
    """
    Perform neighbor joining on a distance matrix D (numpy array, symmetric, zero diag).
    labels: list of taxon names corresponding to rows/cols of D.
    Returns a Newick string (no trailing semicolon).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # D is n x n numpy array
    D = D.astype(np.float64).copy()
    names = list(labels)
    node_names = list(labels)

    while len(node_names) > 3:
        m = len(node_names)
        # Compute row sums (only among active nodes)
        # Active nodes are indexed 0..m-1 in the current D (we maintain D as m x m)
        r = D.sum(axis=1)  # shape (m,)

        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        # We want the minimum Q value over i < j
        # Vectorized: Q = (m-2)*D - r[:,None] - r[None,:]
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
        # D_new[k, new] = 0.5 * (D[i,k] + D[j,k] - D[i,j])
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        # Remove rows/cols i and j, add new row/col
        # Build new D of size (m-1) x (m-1)
        keep = [k for k in range(m) if k != i and k != j]
        new_m = m - 1
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        # Add new node at position new_m (last)
        # Actually we need new_D to be (new_m+1) x (new_m+1)? No.
        # We remove 2 nodes and add 1, so new size = m - 1.
        # The keep list has m-2 elements. We add 1 new node, so total = m-1.
        # Let me redo: new_D should be (m-1) x (m-1).
        # Positions 0..m-3 correspond to keep, position m-2 is the new node.
        new_D = np.zeros((m - 1, m - 1), dtype=np.float64)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        # New node distances
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


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using numpy."""
    n = len(labels)
    # Encode sequences as numpy arrays of integers
    # A=0, C=1, G=2, T=3
    enc = {}
    for lab in labels:
        s = sequences[lab]
        arr = np.zeros(len(s), dtype=np.int8)
        for k, ch in enumerate(s):
            if ch == 'A':
                arr[k] = 0
            elif ch == 'C':
                arr[k] = 1
            elif ch == 'G':
                arr[k] = 2
            elif ch == 'T':
                arr[k] = 3
            else:
                arr[k] = 0  # shouldn't happen
        enc[lab] = arr

    # Compute pairwise Hamming distances
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ai = enc[labels[i]]
            aj = enc[labels[j]]
            # Count differences
            diffs = np.count_nonzero(ai != aj)
            p = diffs / len(ai)
            D[i, j] = D[j, i] = _jc_distance(p)
    return D


def _nj_with_deadline(sequences, labels, deadline):
    """Run NJ with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    D = _compute_distance_matrix(sequences, labels)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    return _neighbor_joining(D, labels)


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
        return result + ";"

    # Fallback: star tree (should not happen given time budget)
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
