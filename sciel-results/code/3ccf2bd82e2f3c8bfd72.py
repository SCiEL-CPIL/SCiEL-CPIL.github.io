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
    
    start_time = time.time()
    deadline = 145.0
    
    # Helper to calculate cut size efficiently
    def calc_cut_size(labels):
        # labels are 0/1.
        # Cut size = 0.5 * sum(A * (labels_i != labels_j))
        # Using matrix multiplication:
        # Let x be spins (-1, 1). 
        # Cut = 0.25 * (sum(A) - x^T A x) is not quite right for unweighted simple graph count?
        # Let's stick to the direct count for accuracy, it's fast enough for n=100.
        # But for speed in local search, we use spin formulation.
        # For final verification, use exact count.
        mask = labels[:, None] != labels[None, :]
        return 0.5 * np.sum(A * mask)

    # Spectral Initialization using Fiedler Vector
    # This provides a strong initial guess closer to the optimal cut than random.
    def spectral_init():
        degrees = A.sum(axis=1)
        L = np.diag(degrees) - A
        # Compute eigenvalues and eigenvectors
        # For n=100, full eigendecomposition is fast
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # eigenvalues are sorted in ascending order by eigh
            fiedler_vec = eigenvectors[:, 1]
            # Partition based on the sign of the Fiedler vector
            labels = (fiedler_vec >= 0).astype(np.int8)
            # Ensure we don't have all 0s or all 1s (unlikely but possible)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, size=n).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n).astype(np.int8)

    # Vectorized Local Search
    # This function performs parallel steepest ascent.
    # In each iteration, it calculates the gain of flipping every vertex simultaneously,
    # flips all vertices with positive gain, and repeats until no positive gains remain.
    # This is much faster than sequential flips for n=100.
    def vectorized_local_search(labels):
        x = labels.copy()
        x[x == 0] = -1
        x[x == 1] = 1
        
        # We iterate until no improvement is found
        # Max iterations safeguard
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            # Calculate the "force" or gain for each vertex
            # Ax = A @ x
            Ax = A @ x
            # Gain for flipping v is -2 * x[v] * Ax[v] in energy terms (minimizing x^T A x)
            # Or simply: if x[v] * Ax[v] > 0, flipping reduces x^T A x (increases cut)
            # Let's check:
            # Current contribution to x^T A x from v: x[v] * sum_j A[v,j] x[j]
            # If we flip x[v] to -x[v], new contribution: -x[v] * sum_j A[v,j] x[j]
            # Change = -2 * x[v] * (Ax)[v]
            # We want to minimize x^T A x. So if Change < 0, we improve.
            # -2 * x[v] * (Ax)[v] < 0  =>  x[v] * (Ax)[v] > 0
            # So, if x[v] * (Ax)[v] > 0, flipping v improves the cut.
            
            gains = x * Ax
            
            # Find vertices that would improve the cut if flipped
            improve_mask = gains > 0
            
            if not np.any(improve_mask):
                break
            
            # Flip all improving vertices simultaneously
            x[improve_mask] = -x[improve_mask]
            
        # Convert back to 0/1
        return (x > 0).astype(np.int8)

    best_labels = None
    best_cut = -1
    
    # Initial best from spectral init + local search
    init_labels = spectral_init()
    init_labels = vectorized_local_search(init_labels)
    init_cut = calc_cut_size(init_labels)
    best_labels = init_labels.copy()
    best_cut = init_cut
    
    # Multi-Start Strategy
    # Since vectorized local search is very fast (likely < 1ms per run for n=100),
    # we can afford many restarts.
    # We mix random starts and perturbed spectral starts.
    
    num_restarts = 500 # High number of restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Strategy: 
        # 1. 50% Random starts
        # 2. 50% Spectral starts (maybe with noise)
        
        if i % 2 == 0:
            # Random start
            init = np.random.randint(0, 2, size=n).astype(np.int8)
        else:
            # Spectral start with slight noise to escape local optima
            spec_init = spectral_init()
            # Flip a small random subset of vertices
            noise_mask = np.random.random(n) < 0.1
            init = spec_init.copy()
            init[noise_mask] = 1 - init[noise_mask]
        
        # Polish with vectorized local search
        labels = vectorized_local_search(init)
        cut_val = calc_cut_size(labels)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels.copy()
            
        # Optional: If we have time, we could try a few more aggressive moves, 
        # but for now, the power of multi-start with fast local search is the key.
        
    return best_labels
# EVOLVE-BLOCK-END
