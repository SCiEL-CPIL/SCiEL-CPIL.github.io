"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via Vectorized Best-Improvement Local Search with Multi-Start.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline guard: 150s wall-clock, use 140s as safety margin
    deadline = time.time() + 140
    
    def get_cut_size(labels):
        # Cut size = sum of edges where labels differ
        # For each edge (i,j), it's in the cut if labels[i] != labels[j]
        # Using: cut = (n/2 * sum(degrees) - sum of (same-side edges))
        # More directly: cut = 0.5 * sum over all i,j of adj[i,j] * (labels[i] != labels[j])
        # But we can compute: for each node, count neighbors on opposite side
        # cut = 0.5 * sum_v (degree_v - same_side_count_v)
        # Let's use: cut = sum over edges where endpoints differ
        # Vectorized: 
        diff = (labels[:, None] != labels[None, :])
        return 0.5 * np.sum(adj * diff)
    
    def local_search(labels, max_iters=50):
        """Vectorized best-improvement local search."""
        labels = labels.copy()
        best_cut = get_cut_size(labels)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gain for flipping each vertex
            # If we flip vertex v, the change in cut is:
            # For each neighbor u of v:
            #   If u was on opposite side (contributing 1 to cut), after flip u is same side (contributes 0): -1
            #   If u was on same side (contributing 0 to cut), after flip u is opposite side (contributes 1): +1
            # So gain_v = (# neighbors on same side) - (# neighbors on opposite side)
            #            = 2 * (# neighbors on same side) - degree_v
            
            # For each vertex v, same_side_count = sum of adj[v, :] where labels == labels[v]
            # opposite_side_count = sum of adj[v, :] where labels != labels[v]
            # gain = same - opposite = 2*same - degree
            
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            degrees = np.sum(adj, axis=1)
            gains = 2 * same_counts - degrees
            
            # Find best vertex to flip
            best_v = np.argmax(gains)
            if gains[best_v] <= 0:
                break
            
            # Flip the best vertex
            labels[best_v] = 1 - labels[best_v]
            best_cut = get_cut_size(labels)
        
        return labels, best_cut
    
    best_labels = None
    best_score = -1
    
    # Multiple random restarts
    rng = np.random.default_rng(42)
    
    # Start with alternating pattern
    starts = []
    starts.append(np.arange(n) % 2)
    
    # Add random starts
    num_random_starts = 20
    for i in range(num_random_starts):
        if time.time() > deadline - 10:
            break
        starts.append(rng.integers(0, 2, size=n))
    
    for start_labels in starts:
        if time.time() > deadline - 5:
            break
        
        result_labels, result_score = local_search(start_labels)
        
        if result_score > best_score:
            best_score = result_score
            best_labels = result_labels
    
    if best_labels is None:
        return np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
