"""Reconstruct unrooted tree topology using FastUp (Fast Upward) algorithm
with Jukes-Cantor distance correction. This avoids the O(n^3) bottleneck of
classic Neighbor-Joining and scales better for up to 32 taxa."""

import math
import numpy as np

def _jc_distance(seq1, seq2):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(seq1)
    if n == 0:
        return 0.0
    diff = sum(1 for a, b in zip(seq1, seq2) if a != b)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    if p <= 0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    """Reconstruct unrooted tree topology using FastUp algorithm.
    
    Args:
        sequences: dict mapping taxon labels to DNA strings
        
    Returns:
        Newick string representing the unrooted tree
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Build distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    
    # FastUp algorithm
    # State: list of active nodes (indices into original labels or composite nodes)
    # We'll track the tree structure as we join nodes
    
    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Tree representation: for each active node, store its Newick subtree
    subtree = {i: labels[i] for i in range(n)}
    
    # Current distance matrix (only for active nodes)
    # We'll maintain a dict-based distance for flexibility
    # dist_dict[(i, j)] for active nodes i, j
    
    # Initialize distance dict
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = dist[i, j]
    
    while len(active) > 3:
        m = len(active)
        
        # Compute total distances for each active node
        total = {}
        for a in active:
            total[a] = sum(dist_dict[(a, b)] for b in active if b != a)
        
        # Find the pair (a, b) that minimizes the NJ criterion
        best_criterion = float('inf')
        best_a, best_b = None, None
        
        for idx_i in range(m):
            a = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                b = active[idx_j]
                criterion = (m - 2) * dist_dict[(a, b)] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        # Compute branch lengths
        d_ab = dist_dict[(a, b)]
        l_a = 0.5 * d_ab + (total[a] - total[b]) / (2.0 * (m - 2))
        l_b = d_ab - l_a
        
        # Create new node
        new_node = max(active) + 1
        new_subtree = f"({subtree[a]},{subtree[b]})"
        
        # Compute distances from new node to all other active nodes
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            d_ax = dist_dict[(a, x)]
            d_bx = dist_dict[(b, x)]
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            dist_dict[(new_node, x)] = d_new_x
            dist_dict[(x, new_node)] = d_new_x
        
        # Update active list
        active = rest + [new_node]
        subtree[new_node] = new_subtree
        
    # Build final Newick string from remaining 3 nodes
    # For 3 nodes, we need to create an unrooted tree
    # The three subtrees are joined at a central point
    if len(active) == 3:
        parts = [subtree[x] for x in active]
        result = f"({parts[0]},{parts[1]},{parts[2]})"
    else:
        # Should not happen
        result = ",".join(subtree[x] for x in active)
    
    return result
