"""Phylogenetic topology recovery via Neighbor Joining with time-guarded fast search."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0
_PER_INSTANCE_BUDGET = 12.0  # seconds per instance as a soft cap


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy."""
    n = len(labels)
    # Convert sequences to integer arrays for fast comparison
    # A=0, C=1, G=2, T=3
    seq_arr = np.zeros((n, len(sequences[labels[0]])), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Vectorized conversion
        for j, ch in enumerate(s):
            if ch == 'A':
                seq_arr[i, j] = 0
            elif ch == 'C':
                seq_arr[i, j] = 1
            elif ch == 'G':
                seq_arr[i, j] = 2
            else:  # T
                seq_arr[i, j] = 3

    # Count differences: for each pair, count positions where they differ
    # Use broadcasting: (n, n, L) would be too big for large L, so do it in chunks
    L = seq_arr.shape[1]
    diff_counts = np.zeros((n, n), dtype=np.int32)
    # Process in chunks of positions to limit memory
    chunk_size = min(L, 500)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk = seq_arr[:, start:end]  # (n, chunk_len)
        # For each pair (i,j), count where chunk[i] != chunk[j]
        # Use broadcasting: chunk[:, None, :] vs chunk[None, :, :]
        # This is (n, n, chunk_len) which for n=32, chunk_len=500 is 32*32*500 = 512k ints, fine
        diff = (chunk[:, None, :] != chunk[None, :, :]).sum(axis=2)
        diff_counts += diff

    p = diff_counts.astype(np.float64) / L
    # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or negative args
    p = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    # Ensure symmetry and zero diagonal
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) * 0.5
    return D


def _neighbor_joining(D, labels):
    """
    Neighbor Joining algorithm.
    D: n x n distance matrix (numpy array)
    labels: list of taxon labels
    Returns Newick string.
    """
    n = len(labels)
    if n <= 2:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        elif n == 1:
            return f"{labels[0]};"
        else:
            return ""

    # Active nodes
    nodes = list(range(n))  # indices into labels
    trees = {i: labels[i] for i in range(n)}
    # Current distance matrix (sparse dict or full array)
    # For efficiency, maintain a dict of active distances
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]

    counter = n  # for new node names

    while len(nodes) > 3:
        if time.time() > _DEADLINE:
            break

        active = nodes
        m = len(active)

        # Compute r_i = sum of distances from i to all other active nodes
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)

        # Find pair (a,b) minimizing (m-2)*d(a,b) - r[a] - r[b]
        best_criterion = float('inf')
        best_a, best_b = active[0], active[1]
        for idx_i in range(m):
            a = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                b = active[idx_j]
                d_ab = dist.get((a, b), 0.0)
                criterion = (m - 2) * d_ab - r[a] - r[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b
        d_ab = dist.get((a, b), 0.0)

        # Compute branch lengths
        # delta = (m-2)
        delta = m - 2
        r_a = r[a]
        r_b = r[b]
        # Branch length from a to new node
        # d_a = 0.5 * d_ab + (r_a - r_b) / (2 * (m - 2))
        if delta != 0:
            d_a = 0.5 * d_ab + (r_a - r_b) / (2.0 * delta)
            d_b = 0.5 * d_ab - (r_a - r_b) / (2.0 * delta)
        else:
            d_a = d_b = 0.5 * d_ab

        # Ensure non-negative
        if d_a < 0:
            d_a = 0.0
        if d_b < 0:
            d_b = 0.0

        # Create new node
        new_node = counter
        counter += 1

        # Build tree string for the joined pair
        tree_a = trees[a]
        tree_b = trees[b]
        # Format with branch lengths
        new_tree = f"({tree_a}:{d_a:.6f},{tree_b}:{d_b:.6f})"
        trees[new_node] = new_tree

        # Update distances: for each remaining x, d(new, x) = 0.5*(d(a,x) + d(b,x) - d(a,b))
        remaining = [x for x in active if x != a and x != b]
        for x in remaining:
            d_ax = dist.get((a, x), 0.0)
            d_bx = dist.get((b, x), 0.0)
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            if d_new_x < 0:
                d_new_x = 0.0
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x

        # Remove old nodes
        nodes = remaining + [new_node]

    # Final tree: connect remaining 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist.get((a, b), 0.0)
        d_ac = dist.get((a, c), 0.0)
        d_bc = dist.get((b, c), 0.0)
        # For 3 taxa, the tree is a star or one pair joined
        # Branch lengths:
        d_a = 0.5 * (d_ab + d_ac - d_bc)
        d_b = 0.5 * (d_ab + d_bc - d_ac)
        d_c = 0.5 * (d_ac + d_bc - d_ab)
        if d_a < 0:
            d_a = 0.0
        if d_b < 0:
            d_b = 0.0
        if d_c < 0:
            d_c = 0.0
        result = f"({trees[a]}:{d_a:.6f},{trees[b]}:{d_b:.6f},{trees[c]}:{d_c:.6f});"
    elif len(nodes) == 2:
        a, b = nodes
        d_ab = dist.get((a, b), 0.0)
        result = f"({trees[a]}:{d_ab/2:.6f},{trees[b]}:{d_ab/2:.6f});"
    elif len(nodes) == 1:
        result = f"{trees[nodes[0]]};"
    else:
        # Fallback: star tree
        result = "(" + ",".join(trees[x] for x in nodes) + ");"

    return result


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns Newick string.
    """
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if n == 1 else ""

    if n == 2:
        return f"({labels[0]},{labels[1]});"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)

    # Time check
    elapsed = time.time() - start_time
    if elapsed > _PER_INSTANCE_BUDGET:
        # Still return the result; NJ is fast enough typically
        pass

    return result
# EVOLVE-BLOCK-END
