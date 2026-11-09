"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing with optimized gain updates.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    # Precompute neighbors for faster iteration
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    
    def get_cut_size(labels):
        mask_0 = (labels == 0)
        mask_1 = (labels == 1)
        return np.sum(adj[np.ix_(mask_0, mask_1)])
    
    # Initialize delta array
    # delta[v] = gain of flipping v
    # delta[v] = 2 * (edges to same side) - degree[v]
    
    def init_delta(labels):
        deltas = np.zeros(n, dtype=int)
        for v in range(n):
            same_side_mask = (labels == labels[v])
            s = np.dot(adj[v], same_side_mask)
            deltas[v] = 2 * s - degrees[v]
        return deltas

    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Initial random assignment
    labels = np.random.randint(0, 2, n)
    current_cut = get_cut_size(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    deltas = init_delta(labels)
    
    # Cooling schedule parameters
    T_start = 1000.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    T = T_start
    
    while time.time() - start_time < time_limit:
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        
        gain = deltas[v]
        
        # Accept if gain > 0 or with probability exp(-gain/T)
        if gain > 0 or random.random() < np.exp(-gain / T):
            # Flip v
            old_label = labels[v]
            new_label = 1 - old_label
            labels[v] = new_label
            current_cut += gain
            
            # Update deltas
            # 1. v's own delta flips sign
            deltas[v] = -gain
            
            # 2. Update neighbors of v
            for u in neighbors[v]:
                # If u and v were same side before, they are opposite now.
                # u's "same side" count decreases by 1 (if edge exists)
                # If u and v were opposite before, they are same now.
                # u's "same side" count increases by 1
                
                if labels[u] == new_label:
                    # u is now same side as v. Before, u was opposite to v.
                    # So edge (u,v) now contributes to u's same-side sum.
                    # delta[u] increases by 2
                    deltas[u] += 2
                else:
                    # u is now opposite side to v. Before, u was same side as v.
                    # So edge (u,v) no longer contributes to u's same-side sum.
                    # delta[u] decreases by 2
                    deltas[u] -= 2
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= cooling_rate
        
        # If T is too low, restart with higher T to avoid getting stuck
        if T < T_end:
            # Reheat
            T = T_start * 0.5
            # Randomize current state to jump out of basin
            labels = np.random.randint(0, 2, n)
            current_cut = get_cut_size(labels)
            deltas = init_delta(labels)
            # Update best if new random is better
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            T = T_start # Full reheat
    
    return best_labels
# EVOLVE-BLOCK-END
