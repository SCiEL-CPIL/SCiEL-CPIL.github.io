"""Spectral initialization + vectorized best-improvement greedy local search for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    """
    Returns a 0/1 label array for each vertex, maximizing the number of edges
    crossing the partition.
    
    Strategy:
    1. Build the graph Laplacian.
    2. Compute the second smallest eigenvector (Fiedler vector) of the Laplacian.
    3. Initialize partition by thresholding the Fiedler vector at its median.
    4. Run vectorized best-improvement greedy local search.
    5. Return the final labels.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Convert adjacency to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure it's a float array for matrix operations
    A = adjacency.astype(float)
    
    # Build Laplacian: L = D - A
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    L = D - A
    
    # Spectral Initialization
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # The first eigenvector corresponds to the smallest eigenvalue (0)
        # The second corresponds to the Fiedler vector
        if eigvecs.shape[1] > 1:
            fiedler = eigvecs[:, 1]
        else:
            fiedler = eigvecs[:, 0]
            
        # Initialize labels based on the Fiedler vector
        # Split at the median to ensure balanced partitions
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        
        # Ensure both partitions are non-empty if possible
        if labels.sum() == 0 or labels.sum() == n:
            # Fallback to alternating
            labels = np.arange(n) % 2
            
    except Exception:
        # Fallback to random initialization
        labels = np.random.randint(0, 2, n)
        
    # Vectorized Best-Improvement Local Search
    # Precompute degrees
    deg = A.sum(axis=1)
    
    # Iterate until no improvement is found
    # Cap iterations to prevent infinite loops in pathological cases, 
    # though convergence is typically fast for n=100.
    for _ in range(1000):
        # Calculate sum of labels of neighbors for each vertex
        # neighbor_labels_sum[i] = sum(labels[j] for j where A[i,j]=1)
        neighbor_labels_sum = A @ labels
        
        # Calculate number of same-label neighbors for each vertex
        # If labels[i] == 0: same = deg[i] - neighbor_labels_sum[i]
        # If labels[i] == 1: same = neighbor_labels_sum[i]
        same_counts = np.where(labels == 0, deg - neighbor_labels_sum, neighbor_labels_sum)
        
        # Calculate gain in cut size if vertex i is flipped
        # Gain = NewSame - OldDiff = Same - (Deg - Same) = 2*Same - Deg
        gains = 2 * same_counts - deg
        
        # Find the vertex with the maximum gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        # If the best possible flip does not improve the cut, we are at a local optimum
        if max_gain <= 0:
            break
        
        # Flip the vertex
        labels[max_gain_idx] = 1 - labels[max_gain_idx]
        
    return labels
# EVOLVE-BLOCK-END
