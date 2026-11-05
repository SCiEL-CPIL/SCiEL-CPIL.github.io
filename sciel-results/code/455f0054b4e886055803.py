"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Max-Cut using Simulated Annealing with multi-restart.
    
    Approach:
    1. Use Simulated Annealing to escape local optima.
    2. Accept moves that decrease cut size with a probability that decreases over time.
    3. Use multiple random restarts to find a global optimum.
    
    For 100-node graphs, this is fast and robust.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    degree = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        return np.sum(adj[labels == 0][:, labels == 1])
    
    def delta_cut(v, labels):
        """
        Calculate change in cut size if vertex v is flipped.
        Delta = 2 * (neighbors in same partition) - degree(v)
        """
        same_partition = np.sum(adj[v] * (labels == labels[v]))
        return 2.0 * same_partition - degree[v]
    
    def simulated_annealing(labels_init, max_iters, T_start, T_end):
        """Run Simulated Annealing starting from labels_init."""
        labels = labels_init.copy()
        current_cut = cut_size(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        # Cooling schedule
        alpha = (T_end / T_start) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            # Pick random vertex
            v = np.random.randint(n)
            
            # Calculate delta
            dc = delta_cut(v, labels)
            
            # Acceptance probability
            if dc > 0:
                # Improvement, always accept
                labels[v] = 1 - labels[v]
                current_cut += dc
            else:
                # Deterioration, accept with prob exp(dc / T)
                T = T_start * (alpha ** i)
                if T < 1e-10:
                    break
                prob = np.exp(dc / T)
                if np.random.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += dc
            
            # Update best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, best_cut
    
    best_labels = None
    best_cut = -1
    
    # Multi-restart Simulated Annealing
    num_restarts = 5
    max_iters = 50000
    T_start = 5.0
    T_end = 0.01
    
    for r in range(num_restarts):
        labels_init = np.random.randint(0, 2, n)
        
        labels_sa, cut_sa = simulated_annealing(labels_init, max_iters, T_start, T_end)
        
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa
            
    return best_labels
# EVOLVE-BLOCK-END
