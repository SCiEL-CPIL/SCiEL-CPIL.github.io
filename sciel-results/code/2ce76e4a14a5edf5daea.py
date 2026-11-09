"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    degrees = adjacency.sum(axis=1)
    
    # Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.9995  # Cooling factor
    num_restarts = 10
    max_steps_per_restart = 5000
    best_labels = np.zeros(n, dtype=np.int8)
    best_score = -1
    
    # Helper to calculate cut size
    def get_cut_size(labels):
        # Sum of edges where labels differ
        # (labels[:, None] != labels[None, :]) * adjacency
        # This creates an n x n matrix, which is fine for n=100
        return np.sum(adjacency * (labels[:, None] != labels[None, :]))
    
    for _ in range(num_restarts):
        # Random initialization
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        current_score = get_cut_size(labels)
        
        T = T_start
        steps = 0
        while T > T_end and steps < max_steps_per_restart:
            # Pick a random vertex to flip
            v = np.random.randint(0, n)
            
            # Calculate delta in cut size if we flip v
            # Current edges from v to opposite side: opp_count
            # Current edges from v to same side: same_count
            # If we flip, same becomes opposite and vice versa.
            # New cut contribution from v: same_count
            # Old cut contribution from v: opp_count
            # Delta = same_count - opp_count
            
            same_count = adjacency[v, labels == labels[v]].sum()
            opp_count = adjacency[v, labels != labels[v]].sum()
            
            delta = same_count - opp_count
            
            if delta > 0:
                # Improvement
                labels[v] = 1 - labels[v]
                current_score += delta
            else:
                # Deterioration
                # Probability of accepting: exp(delta / T)
                # delta is negative
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    labels[v] = 1 - labels[v]
                    current_score += delta
            
            T *= alpha
            steps += 1
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
