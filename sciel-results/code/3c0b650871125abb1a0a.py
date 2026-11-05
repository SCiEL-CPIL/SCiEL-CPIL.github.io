"""
Phylogenetic topology recovery using Neighbor Joining with fast matrix operations
and a deadline guard.
"""
import time
import math
import numpy as np

# Global deadline: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0

def _time_exceeded():
    return time.time() > _DEADLINE

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    # Convert sequences to integer arrays for fast comparison
    # A=0, C=1, G=2, T=3
    seq_list = [labels[i] for i in range(n)]
    seqs = [sequences[seq_list[i]] for i in range(n)]
    
    # Get length
    L = len(seqs[0])
    
    # Convert to numpy arrays
    seq_arrays = []
    for s in seqs:
        arr = np.empty(L, dtype=np.int8)
        for j, c in enumerate(s):
            if c == 'A':
                arr[j] = 0
            elif c == 'C':
                arr[j] = 1
            elif c == 'G':
                arr[j] = 2
            else:  # T
                arr[j] = 3
        seq_arrays.append(arr)
    
    # Compute pairwise differences
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_arrays[i] != seq_arrays[j])
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
    Returns a Newick string.
    D is an n x n numpy array, labels is a list of n strings.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Work on a copy
    dist = D.copy()
    names = list(labels)
    
    # Track the tree structure
    # We'll build the tree by keeping track of which nodes are joined
    # Use a list of active nodes
    active = list(range(n))
    # tree_parts: for each active node index, store the Newick substring
    tree_parts = [names[i] for i in range(n)]
    
    # We need to maintain the distance matrix for active nodes
    # active[i] is the original index, but we need a mapping
    # Let's use a different approach: maintain a list of current nodes
    # and a dictionary mapping current node id to its distances to other current nodes
    
    # Simpler approach: use the standard NJ algorithm
    # Maintain a list of nodes and a distance matrix indexed by position
    
    num_nodes = n
    node_names = list(names)
    # dist_matrix is num_nodes x num_nodes
    dist_matrix = dist.copy()
    
    # tree_substrings: for each current node, the Newick string representing it
    tree_substrings = list(node_names)
    
    while num_nodes > 3:
        if _time_exceeded():
            break
        
        m = num_nodes
        # Compute total distances
        r = np.sum(dist_matrix, axis=1)
        
        # Find the pair that minimizes the NJ criterion
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist_matrix[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        
        # Compute branch lengths
        # delta = (r[i] - r[j]) / (2 * (m - 2))
        delta = (r[i] - r[j]) / (2.0 * (m - 2))
        d_ij = dist_matrix[i, j]
        len_i = 0.5 * d_ij + 0.5 * delta
        len_j = 0.5 * d_ij - 0.5 * delta
        
        # Ensure non-negative
        if len_i < 0:
            len_i = 0.0
        if len_j < 0:
            len_j = 0.0
        
        # Create new node
        new_name = f"({tree_substrings[i]}:{len_i:.6f},{tree_substrings[j]}:{len_j:.6f})"
        
        # Compute distances from new node to all other nodes
        new_distances = np.zeros(m - 2)
        new_active = []
        for k in range(m):
            if k == i or k == j:
                continue
            # Newick's formula: d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
            new_distances[len(new_active)] = 0.5 * (dist_matrix[i, k] + dist_matrix[j, k] - d_ij)
            new_active.append(k)
        
        # Build new distance matrix
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        for idx, k in enumerate(new_active):
            for idx2, k2 in enumerate(new_active):
                new_dist[idx, idx2] = dist_matrix[k, k2]
            new_dist[idx, m - 2] = new_distances[idx]
            new_dist[m - 2, idx] = new_distances[idx]
        
        # Update structures
        tree_substrings = [tree_substrings[k] for k in new_active] + [new_name]
        dist_matrix = new_dist
        num_nodes = m - 1
    
    # Combine remaining nodes
    if num_nodes == 3:
        # For 3 nodes, just make a star tree
        a, b, c = range(3)
        # Compute branch lengths for the star
        # For 3 taxa, the branch lengths are:
        # len_a = 0.5 * (d(a,b) + d(a,c) - d(b,c))
        len_a = 0.5 * (dist_matrix[0, 1] + dist_matrix[0, 2] - dist_matrix[1, 2])
        len_b = 0.5 * (dist_matrix[0, 1] + dist_matrix[1, 2] - dist_matrix[0, 2])
        len_c = 0.5 * (dist_matrix[0, 2] + dist_matrix[1, 2] - dist_matrix[0, 1])
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        if len_c < 0: len_c = 0.0
        result = f"({tree_substrings[0]}:{len_a:.6f},{tree_substrings[1]}:{len_b:.6f},{tree_substrings[2]}:{len_c:.6f})"
    elif num_nodes == 2:
        result = f"({tree_substrings[0]}:{dist_matrix[0,1]/2:.6f},{tree_substrings[1]}:{dist_matrix[0,1]/2:.6f})"
    else:
        result = "(" + ",".join(tree_substrings) + ")"
    
    return result + ";"

def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
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
