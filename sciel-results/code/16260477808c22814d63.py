"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # An edge (i,j) crosses if labels[i] != labels[j]
    # Using: cut = sum over all i<j where labels differ of adj[i,j]
    # Vectorized: (labels[:, None] != labels[None, :]) & adj, then sum and divide by 2 (since each edge counted twice)
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy local improvement
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                # Compute gain from flipping vertex v
                # If v is in group 0, gain = (# neighbors in group 1) - (# neighbors in group 0)
                # If v is in group 1, gain = (# neighbors in group 0) - (# neighbors in group 1)
                if labels[v] == 0:
                    gain = adj[v].sum() - 2 * (adj[v][labels == 0].sum())
                else:
                    gain = 2 * (adj[v][labels == 0].sum()) - adj[v].sum()
                
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        if time.time() > deadline:
            break
        
        # Evaluate this solution
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we never got a chance to run, use alternating start
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels = labels
    
    return best_labels
# EVOLVE-BLOCK-END
