"""Rewritten program for Maximum Cut task with stronger algorithm."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    """
    Solve Maximum Cut using multi-start local search with simulated annealing
    and 2-opt style improvements.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin out of 120s limit

    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)  # Convert to int8 for faster arithmetic
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    best_labels = None
    best_cut = -1.0
    
    def compute_cut(labels):
        """Compute cut size efficiently using matrix operations."""
        # For each vertex, count neighbors with same label
        # same_count[v] = sum of adj[v, labels == labels[v]]
        # cut = (total_degree - sum(same_counts)) / 2
        same_counts = np.array([adj[v][labels == labels[v]].sum() for v in range(n)])
        return (total_degree - same_counts.sum()) / 2.0
    
    def greedy_improve(labels, cut_size):
        """Greedy local search: flip vertices that improve the cut."""
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if time.time() - start_time > DEADLINE:
                    return labels, cut_size
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    cut_size += (same - other)
                    improved = True
        return labels, cut_size
    
    def simulated_annealing(labels, cut_size, iterations=5000):
        """Simulated annealing to escape local optima."""
        current_temp = 1.0
        cooling_rate = 0.9995
        min_temp = 0.01
        
        for _ in range(iterations):
            if time.time() - start_time > DEADLINE:
                break
            if current_temp < min_temp:
                break
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            delta = same - other  # If we flip, cut changes by delta
            
            if delta > 0 or random.random() < np.exp(-delta / max(current_temp, 1e-10)):
                labels[v] = 1 - labels[v]
                cut_size += delta
            
            current_temp *= cooling_rate
        
        return labels, cut_size
    
    def two_opt_improve(labels, cut_size):
        """Try flipping pairs of vertices if it improves the cut."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() - start_time > DEADLINE:
                    return labels, cut_size
                for j in range(i + 1, n):
                    if time.time() - start_time > DEADLINE:
                        return labels, cut_size
                    # Compute delta for flipping both i and j
                    # Before: i has same_i neighbors on same side, other_i on other
                    # After flipping i: i's contribution changes
                    # This is complex, so we just try and compute
                    old_labels = labels.copy()
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    new_cut = compute_cut(labels)
                    if new_cut > cut_size:
                        cut_size = new_cut
                        improved = True
                    else:
                        labels = old_labels
        return labels, cut_size
    
    # Multiple random restarts
    num_restarts = 0
    while time.time() - start_time < DEADLINE:
        num_restarts += 1
        
        # Random initial assignment
        labels = np.random.randint(0, 2, size=n)
        cut_size = compute_cut(labels)
        
        # Greedy improvement
        labels, cut_size = greedy_improve(labels, cut_size)
        
        # Simulated annealing to escape local optima
        labels, cut_size = simulated_annealing(labels, cut_size, iterations=3000)
        
        # Another greedy pass after SA
        labels, cut_size = greedy_improve(labels, cut_size)
        
        # Try 2-opt improvement (limited to avoid timeout)
        if time.time() - start_time < DEADLINE - 5:
            labels, cut_size = two_opt_improve(labels, cut_size)
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()
        
        # Stop if we've done enough restarts or time is running out
        if num_restarts >= 20:
            break
    
    if best_labels is None:
        # Fallback: alternating start
        labels = np.arange(n) % 2
        return labels
    
    return best_labels
# EVOLVE-BLOCK-END
