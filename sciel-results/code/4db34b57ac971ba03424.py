"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Computes a maximum cut partition for the given adjacency matrix.
    Uses spectral initialization followed by local search and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n == 0:
        return np.array([], dtype=int)
    
    # Helper to calculate cut size
    def cut_size(labels):
        # labels: 0/1 array
        # Cut size is sum of edges (i,j) where labels[i] != labels[j]
        # Efficient way: 
        # sum_{i} sum_{j} adj[i,j] * (labels[i] != labels[j]) / 2
        # Or: 0.5 * (sum_all_edges - sum_same_side_edges)
        # Let's use matrix multiplication for speed if possible, but n=100 is small.
        # diff = labels[:, None] != labels[None, :]
        # return np.sum(adj * diff)
        # This creates a 100x100 bool matrix, very fast.
        diff = labels[:, None] != labels[None, :]
        return np.sum(adj * diff)

    # Helper for local search
    def local_search(labels):
        improved = True
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        # Precompute degrees for gain calculation? 
        # Gain of flipping v:
        # new_cut = old_cut + sum_{u in N(v), label[u]==label[v]} 1 - sum_{u in N(v), label[u]!=label[v]} 1
        # gain = (same_side_neighbors - diff_side_neighbors)
        # If gain > 0, flip.
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Randomize order to avoid bias? Or just iterate 0..n-1
            # For deterministic and fast, iterate 0..n-1.
            for v in range(n):
                # Count neighbors with same label and different label
                # neighbors = adj[v]
                # same_side = np.sum(neighbors * (labels == labels[v]))
                # diff_side = np.sum(neighbors * (labels != labels[v]))
                # gain = same_side - diff_side
                # If gain > 0, flipping v increases cut by gain.
                
                # Efficient calculation:
                # labels[v] is 0 or 1.
                # same_side_mask = (labels == labels[v])
                # diff_side_mask = (labels != labels[v])
                
                # We can precompute adj_sum = adj.sum(axis=1) ? No, depends on labels.
                
                # Let's just do the sums. n=100, this is 100 * 100 ops per pass.
                same_side = np.sum(adj[v] * (labels == labels[v]))
                diff_side = np.sum(adj[v] * (labels != labels[v]))
                
                if same_side > diff_side:
                    labels[v] = 1 - labels[v]
                    improved = True
                    # Update best if we want to track, but labels is being modified in place
                    # We check cut at the end or incrementally?
                    # Incremental is better for SA, but for greedy local search, 
                    # we can just flip and continue.
        
        return labels

    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    degrees = np.sum(adj, axis=1)
    D = np.diag(degrees)
    L = D - adj.astype(float)
    
    # Find Fiedler vector (eigenvector of 2nd smallest eigenvalue)
    # For n=100, dense eig is fine and robust.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # eigh returns sorted eigenvalues
        # fiedler_vec is eigenvectors[:, 1]
        fiedler_vec = eigenvectors[:, 1]
        
        # Initialize labels based on sign or median
        # Sign: 0 if < 0, 1 if >= 0
        # Median: 0 if < median, 1 if >= median
        # Sign is standard.
        labels_spec = (fiedler_vec >= 0).astype(int)
        
        # Ensure not all same label (should not happen with Fiedler)
        if np.all(labels_spec == labels_spec[0]):
             # Fallback to random
             labels_spec = np.random.randint(0, 2, n)
            
    except Exception:
        labels_spec = np.random.randint(0, 2, n)

    # Refine spectral solution
    labels_ls_spec = local_search(labels_spec.copy())
    best_labels = labels_ls_spec
    best_cut = cut_size(best_labels)

    # 2. Random Restart (if time permits)
    # Spectral is usually very good, but a random restart might find a better basin.
    if time.time() < deadline - 10.0:
        # Try a few random starts
        for _ in range(3):
            if time.time() >= deadline - 5.0:
                break
            labels_rand = np.random.randint(0, 2, n)
            labels_ls_rand = local_search(labels_rand)
            cut_val = cut_size(labels_ls_rand)
            if cut_val > best_cut:
                best_cut = cut_val
                best_labels = labels_ls_rand

    # 3. Simulated Annealing (if time permits)
    # Use SA to try to escape local minima from the best found so far.
    if time.time() < deadline - 5.0:
        current_labels = best_labels.copy()
        current_cut = best_cut
        
        # SA Parameters
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.95
        T = T_start
        sa_start_time = time.time()
        sa_deadline = deadline - 2.0
        
        while T > T_end and time.time() < sa_deadline:
            # Pick a random vertex to flip
            v = np.random.randint(0, n)
            
            # Calculate gain
            same_side = np.sum(adj[v] * (current_labels == current_labels[v]))
            diff_side = np.sum(adj[v] * (current_labels != current_labels[v]))
            gain = same_side - diff_side
            
            # Accept if gain > 0 or with probability exp(-gain/T) if gain < 0
            if gain > 0 or (np.random.rand() < np.exp(-gain / T)):
                current_labels[v] = 1 - current_labels[v]
                current_cut += gain
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            
            T *= cooling_rate

    return best_labels

# EVOLVE-BLOCK-END
