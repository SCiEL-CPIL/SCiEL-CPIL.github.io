"""Maximum Cut using Spectral Bisection and Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels is 1D array of 0s and 1s
    # Create a matrix where entry (i,j) is 1 if labels[i] != labels[j]
    # Multiply by adjacency and sum, then divide by 2 because edges are counted twice
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2

def _local_search(adj, labels, deadline):
    """
    Fast local search using 1-flip moves to polish the solution.
    Uses vectorized operations for speed.
    """
    n = len(labels)
    labels = labels.copy()
    improved = True
    # Precompute degrees for fast gain calculation? 
    # Gain of flipping v:
    # If v in 0: gain = sum(adj[v, labels==1]) - sum(adj[v, labels==0])
    # If v in 1: gain = sum(adj[v, labels==0]) - sum(adj[v, labels==1])
    
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Vectorized calculation of gains for all vertices
        # sum_to_1[i] = sum(adj[i, labels==1])
        # sum_to_0[i] = sum(adj[i, labels==0])
        
        mask_1 = (labels == 1)
        mask_0 = (labels == 0)
        
        sum_to_1 = adj @ mask_1.astype(np.int8)
        sum_to_0 = adj @ mask_0.astype(np.int8)
        
        # For vertices in 0, gain = sum_to_1 - sum_to_0
        # For vertices in 1, gain = sum_to_0 - sum_to_1
        # We want to flip if gain > 0
        
        gains = np.where(labels == 0, sum_to_1 - sum_to_0, sum_to_0 - sum_to_1)
        
        # Find all vertices that can improve
        to_flip = np.where(gains > 0)[0]
        
        if len(to_flip) > 0:
            # Flip all improving vertices simultaneously? 
            # Standard local search flips one by one, but simultaneous flip of all positive gains
            # is often used in fast heuristics and works well for Max Cut.
            # However, strict 1-flip is safer to guarantee monotonic improvement.
            # Let's do simultaneous flip for speed, as it's a common heuristic variant.
            # Actually, let's stick to flipping all positive gain nodes at once.
            # This is faster and often converges to the same local optimum.
            labels[to_flip] = 1 - labels[to_flip]
            improved = True
            
    return labels

def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    # Safety margin: return 10s before the 120s limit
    deadline = start_time + 110.0
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    try:
        # Eigen-decomposition of L
        # We want the Fiedler vector (2nd smallest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # eigenvectors columns are sorted by eigenvalue
        fiedler = eigenvectors[:, 1]
        
        # Partition based on sign of Fiedler vector
        # Mean-centering is sometimes used, but sign is standard
        labels_spec = (fiedler > 0).astype(np.int8)
        
        # Refine with local search
        labels_spec = _local_search(adj, labels_spec, deadline - 5)
        cut_spec = _cut_size(adj, labels_spec)
        
        if cut_spec > best_cut:
            best_cut = cut_spec
            best_labels = labels_spec.copy()
            
    except Exception:
        pass

    # 2. Random Restarts (if time permits)
    # Spectral is usually very good, but random restarts + local search can sometimes find better optima
    # or handle cases where spectral fails (e.g. disconnected components, though L handles that).
    
    num_random_restarts = 10
    for i in range(num_random_restarts):
        if time.time() > deadline - 5:
            break
            
        labels_rand = np.random.randint(0, 2, size=n)
        labels_rand = _local_search(adj, labels_rand, deadline - 5)
        cut_rand = _cut_size(adj, labels_rand)
        
        if cut_rand > best_cut:
            best_cut = cut_rand
            best_labels = labels_rand.copy()

    # Fallback if something went wrong
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
