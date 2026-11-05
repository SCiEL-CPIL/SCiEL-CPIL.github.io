"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_cut(adjacency_int, labels):
    """Compute cut size given adjacency as int matrix and 0/1 labels."""
    # cut = sum over edges (i,j) where labels differ
    # Efficient: for each vertex, count neighbors with different label
    # cut_size = 0.5 * sum_v sum_u A[v,u] * (labels[v] != labels[u])
    # Using matrix ops: 
    # Let L be the label vector. For each v, same-side neighbors = A @ (labels == labels[v])
    # Actually simpler: cut = 0.5 * (sum of all edges - sum of same-side edges)
    # Or directly: for vertex v, crossing edges = sum_u A[v,u] * (1 - (labels[v] == labels[u]))
    # In numpy: create a matrix where entry (v,u) is 1 if labels differ
    diff_matrix = 1 - (labels[:, None] == labels[None, :])
    cut_size = 0.5 * np.sum(adjacency_int * diff_matrix)
    return cut_size


def _local_search(adjacency_int, labels, max_rounds=20, deadline=None):
    """
    Vectorized local search: repeatedly flip vertices that improve the cut.
    Uses gain computation: gain[v] = (other - same) where
      same = number of neighbors with same label
      other = number of neighbors with different label
    Flipping v changes cut by (other - same). If positive, flip.
    """
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute for each vertex:
        # same[v] = sum of A[v, u] where labels[u] == labels[v]
        # other[v] = sum of A[v, u] where labels[u] != labels[v]
        
        # Create boolean masks
        same_mask = (labels[:, None] == labels[None, :])  # n x n boolean
        # same[v] = sum over u of A[v,u] * same_mask[v,u]
        same = np.sum(adjacency_int * same_mask, axis=1).astype(float)
        other = np.sum(adjacency_int, axis=1) - same
        
        gain = other - same
        
        # Find vertices with positive gain
        flip_mask = gain > 0
        if not np.any(flip_mask):
            break
        
        # Flip all such vertices simultaneously (parallel update)
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels


def _random_labels(n):
    return np.random.randint(0, 2, size=n)


def partition(adjacency):
    """
    Maximum Cut via multi-start vectorized local search with deadline guard.
    
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array)
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = len(adjacency)
    
    # Convert to int for faster arithmetic
    adj_int = adjacency.astype(np.int32)
    
    # Precompute degree
    degrees = np.sum(adj_int, axis=1).astype(np.int32)
    
    best_labels = None
    best_cut = -1
    
    # Multi-start: try several random initializations
    # Plus the alternating start
    num_starts = 0
    
    # Start 1: alternating
    labels = np.arange(n) % 2
    labels = _local_search(adj_int, labels, max_rounds=30, deadline=deadline)
    cut = _calc_cut(adj_int, labels)
    best_labels = labels.copy()
    best_cut = cut
    num_starts += 1
    
    # Additional random starts
    while time.time() < deadline - 5.0:  # Keep 5s buffer
        if num_starts >= 200:  # Cap on starts
            break
        
        labels = _random_labels(n)
        labels = _local_search(adj_int, labels, max_rounds=30, deadline=deadline)
        cut = _calc_cut(adj_int, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        num_starts += 1
    
    # If we have time, do more intensive local search on the best solution
    if time.time() < deadline - 3.0 and best_labels is not None:
        # Polish: more rounds on the best
        polished = _local_search(adj_int, best_labels, max_rounds=50, deadline=deadline)
        cut = _calc_cut(adj_int, polished)
        if cut > best_cut:
            best_labels = polished
            best_cut = cut
    
    return best_labels
# EVOLVE-BLOCK-END
