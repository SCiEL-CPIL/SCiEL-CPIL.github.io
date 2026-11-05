"""Phylogenetic topology recovery using optimized Neighbor Joining with NumPy."""
import time
import math
import numpy as np

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0


def _time_exceeded():
    return time.time() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)
    
    # Convert sequences to integer arrays: A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            seq_arrays[i, j] = char_map.get(c, 0)
    
    # Compute pairwise Hamming distances
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(seq_arrays[i] != seq_arrays[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string (topology only, no branch lengths needed for RF score).
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Work on a copy
    dist = D.copy()
    names = list(labels)
    
    # Track the tree structure: for each current node, store the Newick substring
    tree_parts = list(names)
    active_indices = list(range(n))
    
    num_nodes = n
    while num_nodes > 3:
        if _time_exceeded():
            break
        
        m = num_nodes
        # Compute total distances (row sums)
        r = np.sum(dist, axis=1)
        
        # Find the pair that minimizes the NJ criterion
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        
        # Create new internal node
        new_part = "(" + tree_parts[i] + "," + tree_parts[j] + ")"
        
        # Compute distances from new node to all other nodes
        new_distances = np.zeros(m - 2)
        new_active = []
        for k in range(m):
            if k == i or k == j:
                continue
            # Newick's formula: d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
            new_distances[len(new_active)] = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
            new_active.append(k)
        
        # Build new distance matrix
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        for idx, k in enumerate(new_active):
            for idx2, k2 in enumerate(new_active):
                new_dist[idx, idx2] = dist[k, k2]
            new_dist[idx, m - 2] = new_distances[idx]
            new_dist[m - 2, idx] = new_distances[idx]
        
        # Update structures
        tree_parts = [tree_parts[k] for k in new_active] + [new_part]
        dist = new_dist
        num_nodes = m - 1
    
    # Combine remaining nodes
    if num_nodes == 3:
        result = "(" + ",".join(tree_parts) + ")"
    elif num_nodes == 2:
        result = "(" + ",".join(tree_parts) + ")"
    else:
        result = "(" + ",".join(tree_parts) + ")"
    
    return result + ";"


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string over labels t0..t(n-1).
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)
    
    return result
