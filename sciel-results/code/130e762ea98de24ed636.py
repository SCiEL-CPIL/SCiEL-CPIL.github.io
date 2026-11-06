"""Phylogenetic topology recovery using Distance-based Neighbor Joining with Jukes-Cantor correction."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _neighbor_joining(labels, dist):
    """
    Perform Neighbor Joining on the distance matrix.
    Returns a Newick string representing the unrooted tree topology.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # We'll maintain:
    # - active: list of node IDs currently in the tree
    # - D: distance matrix over all nodes (original + internal)
    # - tree_str: mapping from node ID to its Newick substring
    # - next_id: counter for new internal nodes
    
    active = list(range(n))
    D = dist.copy()
    tree_str = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r_i for each active node
        r = {}
        for i in active:
            r[i] = sum(D[i, j] for j in active if j != i)
        
        # Find pair (i, j) minimizing Q(i,j) = (m-2)*D[i,j] - r[i] - r[j]
        best_q = float('inf')
        best_i, best_j = -1, -1
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q_val = (m - 2) * D[i, j] - r[i] - r[j]
                if q_val < best_q:
                    best_q = q_val
                    best_i, best_j = i, j
        
        # Create new internal node
        new_id = next_id
        next_id += 1
        
        # New tree representation
        tree_str[new_id] = f"({tree_str[best_i]},{tree_str[best_j]})"
        
        # Compute distances from new node to all other active nodes
        others = [x for x in active if x != best_i and x != best_j]
        
        # Ensure D has room for the new node
        if new_id >= D.shape[0]:
            # This shouldn't happen since we pre-allocated, but just in case
            pass
        
        for x in others:
            D[new_id, x] = 0.5 * (D[best_i, x] + D[best_j, x] - D[best_i, best_j])
            D[x, new_id] = D[new_id, x]
        
        # Update active list
        active = [x for x in active if x != best_i and x != best_j]
        active.append(new_id)
    
    # Build final tree from remaining 3 nodes
    reps = [tree_str[x] for x in active]
    return f"({reps[0]},{reps[1]},{reps[2]});"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Pre-allocate D to accommodate internal nodes
    # Max internal nodes = n - 3
    max_nodes = n + (n - 3)
    D = np.zeros((max_nodes, max_nodes))
    D[:n, :n] = dist
    
    # Run Neighbor Joining with pre-allocated D
    # We need to modify _neighbor_joining to accept D and not create new arrays
    # Let's inline the NJ loop here for efficiency
    
    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r_i for each active node
        r = {}
        for i in active:
            r[i] = 0.0
            for j in active:
                if j != i:
                    r[i] += D[i, j]
        
        # Find pair (i, j) minimizing Q(i,j)
        best_q = float('inf')
        best_i, best_j = -1, -1
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q_val = (m - 2) * D[i, j] - r[i] - r[j]
                if q_val < best_q:
                    best_q = q_val
                    best_i, best_j = i, j
        
        # Create new internal node
        new_id = next_id
        next_id += 1
        
        # New tree representation
        tree_str[new_id] = f"({tree_str[best_i]},{tree_str[best_j]})"
        
        # Compute distances from new node to all other active nodes
        others = [x for x in active if x != best_i and x != best_j]
        
        for x in others:
            D[new_id, x] = 0.5 * (D[best_i, x] + D[best_j, x] - D[best_i, best_j])
            D[x, new_id] = D[new_id, x]
        
        # Update active list
        active = [x for x in active if x != best_i and x != best_j]
        active.append(new_id)
    
    # Build final tree from remaining 3 nodes
    reps = [tree_str[x] for x in active]
    return f"({reps[0]},{reps[1]},{reps[2]});"
# EVOLVE-BLOCK-END
