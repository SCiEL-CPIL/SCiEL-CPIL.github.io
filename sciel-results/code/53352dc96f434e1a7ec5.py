"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)

def _local_search(adjacency, labels):
    """Greedy local improvement using efficient gain updates."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    degree = A.sum(axis=1)
    
    # Initialize same_count and gain
    # same_count[v] = number of neighbors of v in the same partition as v
    same_labels = (labels[:, None] == labels[None, :])
    same_count = (A * same_labels).sum(axis=1)
    gain = 2 * same_count - degree
    
    # Greedy sweep
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if gain[v] > 0:
                # Flip v
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update cut size
                # We don't track cut size explicitly here, just gain
                # But we need to update neighbors' gains
                
                for w in range(n):
                    if A[v, w] > 0:
                        if labels[w] == old_label:
                            # w was same as v, now opposite
                            # w's same_count decreases by 1
                            same_count[w] -= 1
                            gain[w] -= 2
                        else:
                            # w was opposite to v, now same
                            # w's same_count increases by 1
                            same_count[w] += 1
                            gain[w] += 2
                
                # Update v's gain
                gain[v] = -gain[v]
                improved = True
    
    return labels

def _spectral_partition(adjacency):
    """Use spectral method to get initial partition via Fiedler vector."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A  # Laplacian
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler >= 0).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, n)
        return labels
    except Exception:
        return np.random.randint(0, 2, n)

def _stochastic_local_search(adjacency, initial_labels, deadline, walk_prob=0.5):
    """WalkSAT-style stochastic local search with efficient gain updates."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    degree = A.sum(axis=1)
    
    labels = initial_labels.copy()
    
    # Initialize same_count and gain
    same_labels = (labels[:, None] == labels[None, :])
    same_count = (A * same_labels).sum(axis=1)
    gain = 2 * same_count - degree
    
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute neighbor lists for faster iteration
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]
    
    while time.time() < deadline:
        # Select vertex to flip
        if np.random.random() < walk_prob:
            v = np.random.randint(n)
        else:
            v = np.argmax(gain)
            # If max gain is negative, we are in a local minimum.
            # We still flip the "least bad" one or random?
            # Standard WalkSAT picks from unsatisfied clauses.
            # Here, if all gains <= 0, any flip worsens or keeps cut same.
            # We proceed with the chosen v.
        
        # Flip v
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        # Update cut size
        current_cut += gain[v]
        
        # Update neighbors
        for w in neighbors[v]:
            if labels[w] == old_label:
                same_count[w] -= 1
                gain[w] -= 2
            else:
                same_count[w] += 1
                gain[w] += 2
        
        # Update v's gain
        gain[v] = -gain[v]
        
        # Update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels, best_cut

def partition(adjacency):
    """Find maximum cut using spectral init, local search, and stochastic local search."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = len(adjacency)
    
    # Strategy 1: Spectral + Local Search
    if time.time() < deadline - 5:
        labels = _spectral_partition(adjacency)
        labels = _local_search(adjacency, labels)
        best_labels = labels.copy()
        best_cut = _cut_size(adjacency, best_labels)
    else:
        best_labels = np.random.randint(0, 2, n)
        best_cut = _cut_size(adjacency, best_labels)
    
    # Strategy 2: Random + Local Search (few restarts if time permits)
    # Since local search is fast, we can try a few random starts
    while time.time() < deadline - 20:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    # Strategy 3: Stochastic Local Search from best found
    # Use remaining time for SLS
    if time.time() < deadline - 2:
        sls_labels, sls_cut = _stochastic_local_search(adjacency, best_labels, deadline - 1, walk_prob=0.5)
        if sls_cut > best_cut:
            best_cut = sls_cut
            best_labels = sls_labels
    
    return best_labels
# EVOLVE-BLOCK-END
