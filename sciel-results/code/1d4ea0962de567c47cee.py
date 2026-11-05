"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing.
    
    Args:
        adjacency: bool matrix of shape (n, n)
        
    Returns:
        labels: array of 0/1 labels per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    # Best solution tracking
    best_labels = None
    best_cut = -1
    
    # Parameters for Simulated Annealing
    n_restarts = 20
    n_iterations = 15000
    T_start = 5.0
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # Random initialization
        if restart == 0:
            # Start with alternating pattern
            labels = np.arange(n) % 2
        else:
            # Random initialization
            labels = rng.integers(0, 2, size=n)
            
        # Compute initial cut size
        # cut = sum of edges between 0s and 1s
        mask0 = labels == 0
        mask1 = labels == 1
        current_cut = adj[np.ix_(mask0, mask1)].sum()
        
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        current_T = T_start
        
        for it in range(n_iterations):
            # Pick a random vertex
            v = rng.integers(0, n)
            
            # Calculate delta in cut size if we flip v
            # Let S be neighbors of v in the same partition
            # Let D be neighbors of v in the different partition
            # Current contribution of v to cut = |D|
            # New contribution of v to cut = |S|
            # Delta = |S| - |D|
            
            # Efficiently get neighbors in same/diff partition
            # labels[v] is the current label
            # same_mask = labels == labels[v]
            # diff_mask = labels != labels[v]
            
            # We can compute this using the adjacency row
            row = adj[v]
            
            # Sum of weights to same side
            # Note: labels are 0 or 1
            # If labels[v] == 0, same side is labels==0
            # If labels[v] == 1, same side is labels==1
            
            # Using boolean indexing
            same_label = labels == labels[v]
            diff_label = labels != labels[v]
            
            same_count = row[same_label].sum()
            diff_count = row[diff_label].sum()
            
            delta = same_count - diff_count
            
            # Acceptance criterion
            if delta > 0:
                # Improvement
                labels[v] = 1 - labels[v]
                current_cut += delta
            else:
                # Deterioration or no change
                # Accept with probability exp(delta / T)
                # If T is very small, exp(delta/T) -> 0 for negative delta
                if current_T > 1e-9:
                    prob = np.exp(delta / current_T)
                    if rng.random() < prob:
                        labels[v] = 1 - labels[v]
                        current_cut += delta
                # else: reject (T is effectively 0)
            
            # Track best local
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
                
            # Update temperature
            current_T *= cooling_rate
            
        # Greedy local search polish
        improved = True
        while improved:
            improved = False
            for v in range(n):
                row = adj[v]
                same_label = best_local_labels == best_local_labels[v]
                diff_label = best_local_labels != best_local_labels[v]
                
                same_count = row[same_label].sum()
                diff_count = row[diff_label].sum()
                
                if same_count > diff_count:
                    best_local_labels[v] = 1 - best_local_labels[v]
                    best_local_cut += (same_count - diff_count)
                    improved = True
        
        # Update global best
        if best_local_cut > best_cut:
            best_cut = best_local_cut
            best_labels = best_local_labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
