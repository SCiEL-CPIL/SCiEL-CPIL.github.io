"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Ensure symmetric
    adj = np.maximum(adj, adj.T)
    # Remove self-loops if any
    np.fill_diagonal(adj, 0)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    def get_cut_size(labels):
        mask = (labels[:, None] != labels[None, :])
        return np.triu(adj * mask, k=1).sum()
    
    rng = np.random.default_rng(42)
    
    # Quantum-inspired Simulated Annealing with multiple restarts
    # We'll run several SA runs with different random seeds and keep the best
    # This helps escape local optima
    
    best_labels_global = None
    best_cut_global = -1
    
    # Number of restarts
    n_restarts = 5
    
    # Time budget per restart
    start_time = time.time()
    total_time_budget = 110.0  # Leave some margin for the 120s limit
    time_per_restart = total_time_budget / n_restarts
    
    for restart in range(n_restarts):
        # Different initialization strategies
        if restart == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif restart == 1:
            # Random
            labels = rng.integers(0, 2, size=n)
        elif restart == 2:
            # Random with bias
            labels = (rng.random(n) < 0.5).astype(np.int8)
        elif restart == 3:
            # Greedy initialization: place each vertex to maximize cut
            labels = np.zeros(n, dtype=np.int8)
            order = rng.permutation(n)
            for v in order:
                # Count neighbors in group 0 vs group 1
                neigh_0 = adj[v][labels == 0].sum()
                neigh_1 = adj[v][labels == 1].sum()
                # If more neighbors in group 0, put v in group 1, and vice versa
                if neigh_0 > neigh_1:
                    labels[v] = 1
                elif neigh_1 > neigh_0:
                    labels[v] = 0
                else:
                    labels[v] = rng.integers(0, 2)
        else:
            # Random
            labels = rng.integers(0, 2, size=n)
        
        current_cut = get_cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # SA parameters
        T_start = 5.0
        T_end = 0.01
        
        # Estimate number of iterations based on time
        # First, do a quick benchmark
        t0 = time.time()
        test_iter = 1000
        for _ in range(test_iter):
            v = rng.integers(0, n)
            cl = labels[v]
            same = adj[v][labels == cl].sum()
            other = adj[v][labels != cl].sum()
        t1 = time.time()
        time_per_iter = (t1 - t0) / test_iter
        
        # Target time for this restart
        target_time = time_per_restart * 0.9  # Leave 10% margin
        iterations = int(target_time / time_per_iter)
        iterations = max(iterations, 10000)
        iterations = min(iterations, 500000)  # Cap at 500k
        
        # Geometric cooling
        cool_factor = (T_end / T_start) ** (1.0 / iterations)
        
        # Run SA
        for i in range(iterations):
            v = rng.integers(0, n)
            
            current_label = labels[v]
            same = adj[v][labels == current_label].sum()
            other = adj[v][labels != current_label].sum()
            
            delta = same - other
            
            if delta > 0:
                labels[v] = 1 - current_label
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                T = T_start * (cool_factor ** i)
                if T > 0.01:  # Only accept worsening moves if temperature is meaningful
                    if rng.random() < np.exp(delta / T):
                        labels[v] = 1 - current_label
                        current_cut += delta
        
        # Final local search polish
        for _ in range(5):
            improved = False
            for v in range(n):
                current_label = labels[v]
                same = adj[v][labels == current_label].sum()
                other = adj[v][labels != current_label].sum()
                if same > other:
                    labels[v] = 1 - current_label
                    current_cut += (same - other)
                    improved = True
            if not improved:
                break
        
        if current_cut > best_cut:
            best_labels = labels.copy()
            best_cut = current_cut
        
        if best_cut > best_cut_global:
            best_cut_global = best_cut
            best_labels_global = best_labels.copy()
        
        # Check if we're running out of time
        elapsed = time.time() - start_time
        if elapsed > total_time_budget * 0.95:
            break
    
    return best_labels_global
# EVOLVE-BLOCK-END
