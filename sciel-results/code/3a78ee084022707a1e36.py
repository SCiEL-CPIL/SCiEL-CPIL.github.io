"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Ensure symmetric and remove self-loops
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # Helper to calculate cut size
    def get_cut_size(labels):
        # Cut size = 0.5 * sum_{i,j} A[i,j] * [x_i != x_j]
        # Since A is symmetric, sum_{i<j} A[i,j] * [x_i != x_j] is the standard cut size.
        mask = (labels[:, None] != labels[None, :])
        return np.triu(adj * mask, k=1).sum()

    best_labels = None
    best_cut = -1
    
    # Time budget: 110 seconds to be safe
    time_budget = 110.0
    start_time = time.time()
    
    # Number of restarts depends on time available per restart.
    # Let's aim for short, intense runs.
    # Each restart: ~5000-10000 iterations?
    # With n=100, one iteration is very fast (~microseconds in Py if optimized).
    # Let's try 20 restarts.
    num_restarts = 20
    
    for restart in range(num_restarts):
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Initialize random labels
        labels = np.random.randint(0, 2, n)
        
        # Calculate initial cut
        current_cut = get_cut_size(labels)
        
        # SA Parameters
        # Start temperature high enough to accept most moves initially
        T_start = 10.0
        # End temperature low enough to freeze
        T_end = 0.1
        # Cooling rate alpha
        alpha = 0.995
        
        T = T_start
        
        # Run SA for a fixed number of iterations or until time slice expires
        # Let's allocate ~5 seconds per restart max, or fixed iterations.
        # 10,000 iterations per restart is safe and fast.
        iterations = 10000
        
        for _ in range(iterations):
            if time.time() - start_time > time_budget:
                break
                
            v = np.random.randint(0, n)
            current_label = labels[v]
            
            # Calculate delta efficiently
            # same = number of neighbors with same label
            # other = number of neighbors with different label
            # delta = same - other
            # Note: degrees[v] = same + other
            # So other = degrees[v] - same
            # delta = same - (degrees[v] - same) = 2*same - degrees[v]
            
            # Count neighbors with same label
            # adj[v] is the row of v. labels is the vector.
            # We want sum of adj[v][i] where labels[i] == labels[v]
            # This can be done as: np.dot(adj[v], (labels == current_label).astype(np.int8))
            # Or simpler:
            same = np.sum(adj[v][labels == current_label])
            
            delta = 2 * same - degrees[v]
            
            if delta > 0:
                labels[v] = 1 - current_label
                current_cut += delta
            else:
                # Accept with probability exp(delta / T)
                # Optimization: if T is very small, exp(delta/T) is ~0
                if T > 1e-6:
                    if np.random.random() < np.exp(delta / T):
                        labels[v] = 1 - current_label
                        current_cut += delta
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
        
        # Track best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        # Final greedy polish on the best found so far? Or just the current one?
        # Let's do a quick greedy polish on the current labels to potentially improve
        # before comparing, or just compare. Greedy polish is cheap (n * few passes).
        for _ in range(5):
            improved = False
            for v in range(n):
                current_label = labels[v]
                same = np.sum(adj[v][labels == current_label])
                delta = 2 * same - degrees[v]
                if delta > 0:
                    labels[v] = 1 - current_label
                    current_cut += delta
                    improved = True
            if not improved:
                break
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
