"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition efficiently."""
    # For binary labels, cut size = sum of adjacency[i,j] where labels[i] != labels[j]
    # Using vectorized boolean comparison
    diff_matrix = (labels[:, None] != labels[None, :])
    return np.sum(adjacency * diff_matrix)

def _local_improvement(labels, adjacency, deadline):
    """Perform local improvement until no more flips improve the cut."""
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        # Randomize the order of vertices to avoid bias
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            
            # Count same-side and other-side neighbors
            same = adjacency[v, labels == labels[v]].sum()
            other = adjacency[v, labels != labels[v]].sum()
            
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels

def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    # Use the structural idea of alternating start (from inspiration) combined
    # with the refinement machinery of multiple restarts and randomization (from current)
    
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 4 == 0:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        elif restart % 4 == 1:
            # Alternating (from inspiration)
            labels = np.arange(n) % 2
        elif restart % 4 == 2:
            # Random with slight bias
            labels = np.random.randint(0, 2, n)
            # Flip a random subset
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        else:
            # Start from alternating but with random perturbation
            labels = np.arange(n) % 2
            # Randomly flip some vertices
            flip_mask = np.random.random(n) < 0.2
            labels[flip_mask] = 1 - labels[flip_mask]
        
        # Local improvement
        labels = _local_improvement(labels.copy(), adjacency, deadline)
        
        # Compute cut size
        cut = _cut_size(labels, adjacency)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final check: if we never found a valid result, use a simple default
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
