"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    def cut_size(labels):
        """Compute cut size using matrix multiplication."""
        # For each vertex, count edges to opposite side
        # labels: 0/1 array
        # edge crosses if labels[i] != labels[j] and adj[i,j] == 1
        # sum over all i,j of adj[i,j] * (labels[i] != labels[j]) / 2
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int((adj * diff).sum() // 2)
    
    def greedy_improve(labels, adj, deadline):
        """Greedy local search: flip vertices if it improves cut."""
        n = len(labels)
        labels = labels.copy()
        
        # Precompute for efficiency
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Random order to break symmetry
            order = np.random.permutation(n)
            for v in order:
                # Count neighbors in same group vs other group
                same_side = labels == labels[v]
                other_side = labels != labels[v]
                
                same_count = adj[v, same_side].sum()
                other_count = adj[v, other_side].sum()
                
                # If flipping improves cut
                # Current contribution of v to cut: other_count
                # After flip: same_count
                if same_count > other_count:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        return labels
    
    # Multiple random restarts
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy improvement
        labels = greedy_improve(labels, adj, deadline)
        
        # Evaluate
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # If we didn't find any valid solution (shouldn't happen), use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
