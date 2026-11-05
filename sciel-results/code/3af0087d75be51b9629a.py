"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Max-Cut using spectral initialization + multi-restart local search.
    
    Approach:
    1. Compute the Laplacian eigenvector corresponding to the smallest non-zero eigenvalue
       (Fiedler vector) to get a high-quality initial partition.
    2. Apply greedy local improvement (flip vertices that increase cut size).
    3. Use multiple random restarts with different initializations to avoid local optima.
    
    For 100-node graphs, this is fast and typically finds near-optimal solutions.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    def local_search(labels):
        """Greedy local improvement: flip vertex if it increases cut size."""
        labels = labels.copy()
        improved = True
        while improved:
            improved = False
            for v in range(n):
                # Count neighbors in same partition vs other partition
                same = np.sum(adj[v][labels == labels[v]])
                other = np.sum(adj[v][labels != labels[v]])
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        return np.sum(adj[labels == 0][:, labels == 1])
    
    def spectral_init():
        """Use Fiedler vector for initial partition."""
        # Compute Laplacian
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj
        # Compute eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
            # (assuming connected graph; if not, still works reasonably)
            fiedler = eigvecs[:, 1]
            # Partition based on sign of Fiedler vector
            labels = (fiedler >= 0).astype(np.int64)
            # Ensure both partitions are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.random.randint(0, 2, n)
            return labels
        except Exception:
            return np.random.randint(0, 2, n)
    
    best_labels = None
    best_cut = -1
    
    # First try spectral initialization
    labels_spec = spectral_init()
    labels_improved = local_search(labels_spec)
    cut_val = cut_size(labels_improved)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = labels_improved
    
    # Multiple random restarts with different initializations
    # For 100 nodes, we can afford several restarts
    num_restarts = 10
    for r in range(num_restarts):
        # Random initial partition
        labels_rand = np.random.randint(0, 2, n)
        # Ensure both partitions are non-empty
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
            labels_rand[0] = 0
            labels_rand[1] = 1
        labels_improved = local_search(labels_rand)
        cut_val = cut_size(labels_improved)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels_improved
    
    return best_labels
# EVOLVE-BLOCK-END
