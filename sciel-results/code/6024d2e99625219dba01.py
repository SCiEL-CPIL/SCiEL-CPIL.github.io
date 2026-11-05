"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency)
    n = len(adj)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute degree sums
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        # Number of edges where labels differ
        # labels is 0/1 array
        # diff is 1 if labels differ
        diff = (labels[:, None] != labels[None, :])
        return np.sum(adj * diff)
        
    def flip_gain(v, labels):
        # Gain in cut size if we flip vertex v
        # Current contribution of v to cut: edges to opposite side
        # New contribution: edges to same side (which become opposite)
        # Let S be set of neighbors of v
        # Let S_same be neighbors with same label
        # Let S_diff be neighbors with diff label
        # Current cut edges from v = |S_diff|
        # New cut edges from v = |S_same|
        # Gain = |S_same| - |S_diff|
        # Note: |S_same| + |S_diff| = degree(v)
        # Gain = |S_same| - (degree(v) - |S_same|) = 2*|S_same| - degree(v)
        
        same_label = labels == labels[v]
        num_same = np.sum(adj[v, same_label])
        return 2 * num_same - degrees[v]

    best_labels = None
    best_score = -1
    
    # Multi-start strategy
    # We will run multiple restarts if time permits
    # Each restart: initialize randomly, then simulated annealing
    
    # Number of restarts depends on time, but we'll just loop until deadline
    # Let's estimate cost of one SA run. For n=100, one pass is O(n^2) or O(E).
    # SA might take ~1000-10000 steps per run.
    
    while time.time() < deadline - 2.0: # Leave 2s for final check
        
        # Initialize labels randomly
        labels = np.random.randint(0, 2, size=n)
        
        # Simulated Annealing Parameters
        # Start temperature high enough to accept bad moves
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        
        current_labels = labels.copy()
        current_score = cut_size(current_labels)
        
        # Track best within this run
        run_best_labels = current_labels.copy()
        run_best_score = current_score
        
        # SA Loop
        while T > T_min:
            if time.time() > deadline - 2.0:
                break
                
            # Pick random vertex
            v = np.random.randint(0, n)
            
            gain = flip_gain(v, current_labels)
            
            # Acceptance probability
            if gain >= 0:
                accept = True
            else:
                # P = exp(gain / T)
                # gain is negative
                accept = np.random.rand() < np.exp(gain / T)
                
            if accept:
                current_labels[v] = 1 - current_labels[v]
                current_score += gain
                
                if current_score > run_best_score:
                    run_best_score = current_score
                    run_best_labels = current_labels.copy()
            
            T *= cooling_rate
            
        # End of SA run for this restart
        if run_best_score > best_score:
            best_score = run_best_score
            best_labels = run_best_labels.copy()
            
        # If we've used too much time for a full run, break
        # A rough heuristic: if n is small, many restarts. If n large, fewer.
        # For n=100, one SA run is fast.
        
    return best_labels
# EVOLVE-BLOCK-END
