"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Max-Cut using Spectral Partitioning followed by Local Refinement.
    
    Approach:
    1. Compute the Fiedler vector (eigenvector of the second smallest eigenvalue 
       of the graph Laplacian).
    2. Use the sign of the Fiedler vector to initialize the partition.
    3. Refine the partition using local search (flip vertices that improve the cut).
    
    This approach leverages global graph structure for initialization, 
    unlike random initialization in Simulated Annealing.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # 1. Spectral Initialization
    # Build Laplacian L = D - A
    degree = adj.sum(axis=1)
    
    # L = D - A
    L = np.diag(degree) - adj
    
    # Find eigenvalues and eigenvectors
    # We want the second smallest eigenvalue's eigenvector
    # np.linalg.eigh returns eigenvalues in ascending order
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The smallest eigenvalue is 0 (constant vector).
        # The second smallest is the Fiedler vector.
        fiedler_vector = eigenvectors[:, 1]
    except np.linalg.LinAlgError:
        # Fallback to random if spectral fails (unlikely for symmetric L)
        fiedler_vector = np.random.randn(n)
        
    # Initial labels based on sign of Fiedler vector
    labels = (fiedler_vector > 0).astype(int)
    
    # 2. Local Refinement (Hill Climbing)
    # Iterate until no improvement or max iterations
    # Efficient delta calculation:
    # Delta(v) = 2 * (sum of edges to same side) - degree(v)
    # If Delta(v) > 0, flipping v increases cut size.
    
    def get_cut_size(labels):
        return np.sum(adj[labels == 0][:, labels == 1])
    
    current_cut = get_cut_size(labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Simple local search: sweep through vertices and flip if beneficial
    # Repeat until no change
    max_passes = 100
    for _ in range(max_passes):
        improved = False
        # Randomize order to avoid bias
        order = np.random.permutation(n)
        for v in order:
            # Calculate change in cut if we flip v
            # Neighbors in same partition: adj[v] * (labels == labels[v])
            same_side_neighbors = np.sum(adj[v] * (labels == labels[v]))
            # Delta = 2 * same_side - degree
            delta = 2.0 * same_side_neighbors - degree[v]
            
            if delta > 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                improved = True
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        if not improved:
            break
            
    return best_labels
# EVOLVE-BLOCK-END
