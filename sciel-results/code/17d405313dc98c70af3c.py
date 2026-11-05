"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Max-Cut using Spectral Partitioning with Greedy Refinement.
    
    Approach:
    1. Compute the graph Laplacian.
    2. Use the Fiedler vector (and other low-frequency eigenvectors) to initialize partitions.
    3. Refine each initialization using a deterministic greedy local search (hill climbing).
    4. Return the best partition found.
    
    This leverages the global structure of the graph via spectral methods,
    distinct from stochastic local search like Simulated Annealing.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])

    # 1. Compute Laplacian L = D - A
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    # 2. Compute eigenvalues and eigenvectors
    # For n=100, this is very fast
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # We want the smallest non-zero eigenvalues.
    # The smallest is always 0 (constant vector).
    # We will try the next few eigenvectors as initializations.
    num_initials = min(5, n - 1)
    
    best_labels = None
    best_cut = -1.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        mask = (labels[:, None] != labels[None, :])
        return np.sum(adj * mask) / 2.0
        
    def local_search(labels):
        """
        Greedy local search: Flip a vertex if it increases the cut size.
        Repeat until no single flip improves the cut.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        improved = True
        while improved:
            improved = False
            for v in range(n):
                same_labels = (labels == labels[v])
                sum_same = np.sum(adj[v] * same_labels)
                delta = 2.0 * sum_same - degree[v]
                
                if delta > 1e-10:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
                    
        return labels, current_cut

    for k in range(1, num_initials + 1):
        vec = eigvecs[:, k]
        
        median_val = np.median(vec)
        labels_init = (vec > median_val).astype(int)
        
        if len(np.unique(labels_init)) < 2:
            labels_init = np.random.randint(0, 2, n)
            
        labels_ls, cut_ls = local_search(labels_init)
        
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls
            
    if best_labels is None:
        labels_init = np.random.randint(0, 2, n)
        best_labels, best_cut = local_search(labels_init)
        
    return best_labels
# EVOLVE-BLOCK-END
