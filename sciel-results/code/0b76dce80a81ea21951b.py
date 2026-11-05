"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.int8)
    
    # Deadline guard: The evaluator kills after 150s. 
    # We aim to finish well before that to be safe, say 140s.
    start_time = time.time()
    deadline = 140.0
    
    # Initialize labels randomly (0 or 1)
    labels = np.random.randint(0, 2, size=n)
    
    # Helper to calculate cut size
    # Cut size = sum of edges where labels differ
    # Efficient calculation: 
    # Let x_v be 1 if label is 1, 0 if label is 0.
    # Actually, let's stick to 0/1 labels as required.
    # cut = sum_{i<j} A[i][j] * (1 if labels[i] != labels[j] else 0)
    
    # A faster way to compute cut size incrementally:
    # If we flip vertex v, the change in cut size is:
    # gain = (number of neighbors in same set) - (number of neighbors in other set)
    # If we flip, we gain the edges to other set and lose edges to same set.
    # So delta_cut = (neighbors in other) - (neighbors in same)
    
    def calc_cut_size(labels):
        # This is O(n^2) but only needed for initialization and final check maybe?
        # No, we need it for the first step.
        # For n=100, n^2 = 10,000 operations. Very fast.
        # We can compute it once at start.
        mask = labels[:, None] != labels[None, :]
        return np.sum(A * mask)
    
    current_cut = calc_cut_size(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Simulated Annealing Parameters
    # Initial temperature: High enough to accept most bad moves initially
    T = 10.0 
    # Cooling factor
    alpha = 0.99
    # Minimum temperature
    T_min = 0.01
    
    # We will iterate until time runs out
    # Each step: pick a random vertex, compute delta, decide to flip
    
    while time.time() < deadline:
        # Pick a random vertex
        v = np.random.randint(0, n)
        
        # Compute change in cut size if we flip v
        # neighbors_same = sum(A[v][labels == labels[v]])
        # neighbors_other = sum(A[v][labels != labels[v]])
        # delta = neighbors_other - neighbors_same
        
        # Efficient calculation using numpy
        # labels is 0/1 array
        # A[v] is the row of the adjacency matrix
        # We want sum of A[v] where labels != labels[v] minus sum where labels == labels[v]
        
        # Let's precompute degrees or just use the array
        # neighbors_other = np.sum(A[v][labels != labels[v]])
        # neighbors_same = np.sum(A[v][labels == labels[v]])
        
        # Even faster:
        # total_degree = A[v].sum()
        # neighbors_same = np.sum(A[v][labels == labels[v]])
        # neighbors_other = total_degree - neighbors_same
        # delta = neighbors_other - neighbors_same
        
        # But we can do it in one line with boolean indexing
        # Note: A is int8, labels is int64 usually.
        
        same_mask = (labels == labels[v])
        other_mask = ~same_mask
        
        # To avoid creating large temporary arrays, we can use:
        # sum_A_v = A[v].sum()
        # sum_same = np.dot(A[v], same_mask) # This works if A[v] is 1D and same_mask is 1D bool
        # Actually np.sum(A[v] * same_mask) is fine for n=100
        
        sum_same = np.sum(A[v][same_mask])
        sum_other = np.sum(A[v][other_mask])
        
        delta = sum_other - sum_same
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # delta is negative (or zero)
            # p = exp(delta / T)
            # If T is very small, exp is 0.
            if T > T_min:
                p = np.exp(delta / T)
                accept = np.random.rand() < p
            else:
                accept = False
        
        if accept:
            labels[v] = 1 - labels[v]
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Cool down
        T *= alpha
        
        # If T is too low, we might want to reheat or restart?
        # Simple SA just keeps going. But if T < T_min, we effectively do greedy.
        # To maintain diversity, we can reset T occasionally or just let it go greedy.
        # Given the time budget, letting it go greedy at the end is good for polishing.
        
    return best_labels
# EVOLVE-BLOCK-END
