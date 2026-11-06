"""Spectral partitioning with iterated local search refinement for Maximum Cut."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array if needed
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Build graph Laplacian: L = D - A
    D = np.diag(degrees)
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to simple alternating pattern
        return np.arange(n) % 2
    
    # The Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
    # eigh returns eigenvalues in ascending order, so index 1 is the second smallest
    fiedler_vector = eigenvectors[:, 1]
    
    # Create initial partition: vertices with positive Fiedler values go to 0, negative to 1
    labels = np.where(fiedler_vector >= 0, 0, 1).astype(int)
    
    def cut_size(labels):
        # Efficient calculation of cut size
        # Cut size = sum of edges where labels differ
        # We can compute this as: 0.5 * sum_{i,j} A[i,j] * (1 if labels[i] != labels[j] else 0)
        # Or simpler: for each i, sum A[i,j] where labels[j] != labels[i]
        # Then divide by 2 because each edge is counted twice.
        # Let's use matrix multiplication for speed if possible, or just loop for n=100.
        # n=100 is small, loop is fine.
        total = 0
        for i in range(n):
            if labels[i] == 0:
                total += A[i][labels == 1].sum()
            else:
                total += A[i][labels == 0].sum()
        return total / 2.0

    def local_search(labels):
        """Perform 1-flip local search until convergence."""
        improved = True
        # Use a random order to avoid bias
        indices = np.arange(n)
        
        while improved:
            improved = False
            np.random.shuffle(indices)
            for v in indices:
                # Count edges going to same side vs other side
                current_label = labels[v]
                other_label = 1 - current_label
                
                # Number of edges from v to other side (crossing)
                edges_crossing = A[v][labels == other_label].sum()
                # Number of edges from v to same side (non-crossing)
                edges_same = A[v][labels == current_label].sum()
                
                # If flipping v increases the cut (edges_same > edges_crossing), flip it
                # Gain = edges_same - edges_crossing
                if edges_same > edges_crossing:
                    labels[v] = other_label
                    improved = True
        
        return labels

    # Initial local search on spectral partition
    current_labels = local_search(labels.copy())
    best_cut = cut_size(current_labels)
    best_labels = current_labels.copy()
    
    # Iterated Local Search: perturb and restart
    num_restarts = 20
    for _ in range(num_restarts):
        # Perturb the current best solution
        perturbed_labels = best_labels.copy()
        # Flip a few random nodes (e.g., 1 to 5)
        k = np.random.randint(1, 6)
        if k > 0:
            flip_indices = np.random.choice(n, k, replace=False)
            perturbed_labels[flip_indices] = 1 - perturbed_labels[flip_indices]
        
        # Run local search on perturbed solution
        refined_labels = local_search(perturbed_labels)
        
        # Check if it's better
        new_cut = cut_size(refined_labels)
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = refined_labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
