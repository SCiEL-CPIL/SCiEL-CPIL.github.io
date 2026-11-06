import numpy as np

def _run_local_search(labels, A, deg):
    """
    Runs best-improvement greedy local search starting from initial labels.
    Returns the final labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees if not passed, but we pass it.
    # A is float matrix, deg is array of degrees.
    
    # First-improvement greedy local search
    improved = True
    while improved:
        improved = False
        neighbor_labels_sum = A @ labels
        same_counts = np.where(labels == 0, deg - neighbor_labels_sum, neighbor_labels_sum)
        gains = 2 * same_counts - deg
        # Find first vertex with positive gain
        positive_gains = np.where(gains > 0)[0]
        if len(positive_gains) == 0:
            break
        # Pick the vertex with the maximum gain among those with positive gain
        best_idx = positive_gains[np.argmax(gains[positive_gains])]
        labels[best_idx] = 1 - labels[best_idx]
        improved = True
        
    # Calculate final cut size
    # Cut size = sum of edges (i,j) where labels[i] != labels[j]
    # This is 0.5 * sum_{i,j} A[i,j] * (1 - (2*labels[i]-1)*(2*labels[j]-1)) ? 
    # Simpler: 
    # Let s_i = 1 if label 0, -1 if label 1? No, let's stick to 0/1.
    # Cut = sum_{i<j} A[i,j] * (labels[i] != labels[j])
    # Vectorized: 
    # labels_diff = 1 - (labels[:, None] == labels[None, :])
    # cut = np.sum(A * labels_diff) / 2 # Since A is symmetric and we count each edge twice in matrix mult?
    # Actually, A is the adjacency matrix.
    # cut = 0.5 * (n * sum(deg) - sum(A @ (2*labels - 1) ** 2)) ? No.
    
    # Simple calculation:
    # cut = np.sum(A * (labels[:, None] != labels[None, :])) / 2
    # This creates an n x n boolean matrix. For n=100, it's fine.
    cut_size = np.sum(A * (labels[:, None] != labels[None, :])) / 2.0
    
    return labels, cut_size

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    A = adjacency.astype(float)
    deg = A.sum(axis=1)
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    try:
        L = np.diag(deg) - A
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        median_val = np.median(fiedler)
        init_labels = (fiedler > median_val).astype(int)
        if init_labels.sum() == 0 or init_labels.sum() == n:
            init_labels = np.arange(n) % 2
    except Exception:
        init_labels = np.random.randint(0, 2, n)
        
    # Run local search on spectral init
    labels, cut = _run_local_search(init_labels, A, deg)
    best_labels = labels
    best_cut = cut
    
    # 2. Random Restarts
    # Do more random restarts since n=100 is small and first-improvement is fast
    num_restarts = 30
    for _ in range(num_restarts):
        random_labels = np.random.randint(0, 2, n)
        labels, cut = _run_local_search(random_labels, A, deg)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
            
    return best_labels
