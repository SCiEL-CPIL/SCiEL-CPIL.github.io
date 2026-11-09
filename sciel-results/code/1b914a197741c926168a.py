"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances with NumPy vectorization.'''
import math
import time
import numpy as np

_DEADLINE = time.time() + 260.0  # Safety margin before 270s kill


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy."""
    n = len(labels)
    seq_array = np.array([list(sequences[lab]) for lab in labels])
    # Convert to one-hot: shape (n, L, 4)
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    numeric = np.zeros((n, seq_array.shape[1]), dtype=np.int8)
    for i in range(n):
        for j in range(seq_array.shape[1]):
            numeric[i, j] = bases[seq_array[i, j]]
    
    # Pairwise differences
    # For each pair, count mismatches
    diff = np.abs(numeric[:, None, :] - numeric[None, :, :])
    # Mismatch where values differ (not both same)
    mismatch = (diff > 0).astype(np.float64)
    L = seq_array.shape[1]
    p = mismatch.sum(axis=2) / L
    # JC correction
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    d = np.where(np.isfinite(d), d, 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_newick(dist, labels):
    """Neighbour joining algorithm. dist is n x n matrix, labels is list of names."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Work with a mutable copy
    d = dist.copy()
    names = list(labels)
    node_map = {i: labels[i] for i in range(n)}
    active = list(range(n))
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        row_sums = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    row_sums[i] += d[active[i], active[j]]
        
        # Find best pair
        best_val = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                ai, aj = active[i], active[j]
                val = (m - 2) * d[ai, aj] - row_sums[i] - row_sums[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        ai = active[best_i]
        aj = active[best_j]
        
        # Create new node
        new_label = f"({node_map[ai]},{node_map[aj]})"
        new_node = next_id
        next_id += 1
        
        # Compute distances from new node to all others
        for k in range(m):
            ak = active[k]
            if ak == ai or ak == aj:
                continue
            d[new_node, ak] = 0.5 * (d[ai, ak] + d[aj, ak] - d[ai, aj])
            d[ak, new_node] = d[new_node, ak]
        
        # Remove old nodes, add new
        active = [a for a in active if a != ai and a != aj]
        active.append(new_node)
        node_map[new_node] = new_label
    
    # Final 3 nodes: connect them
    remaining = active
    if len(remaining) == 3:
        a, b, c = remaining
        return f"({node_map[a]},{node_map[b]},{node_map[c]})"
    elif len(remaining) == 2:
        a, b = remaining
        return f"({node_map[a]},{node_map[b]})"
    else:
        return node_map[remaining[0]]


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        elif n == 1:
            return labels[0] + ";"
        else:
            return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    try:
        result = _nj_newick(d, labels)
    except Exception:
        # Fallback on error
        result = "(" + ",".join(labels) + ");"
    
    return result + ";"
# EVOLVE-BLOCK-END
