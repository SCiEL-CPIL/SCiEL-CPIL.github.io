"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition efficiently."""
    diff_matrix = (labels[:, None] != labels[None, :])
    return np.sum(adjacency * diff_matrix)

def _local_improvement(labels, adjacency, deadline):
    """Perform local improvement until no more flips improve the cut."""
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            
            same = adjacency[v, labels == labels[v]].sum()
            other = adjacency[v, labels != labels[v]].sum()
            
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels

def _second_order_improvement(labels, adjacency, deadline):
    """Try pairs of flips that individually don't help but together do."""
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        num_pairs = min(500, n * (n - 1) // 2)
        for _ in range(num_pairs):
            if time.time() > deadline:
                break
            
            i, j = np.random.choice(n, 2, replace=False)
            if labels[i] == labels[j]:
                continue
            
            li = labels[i]
            lj = labels[j]
            
            old_cut = _cut_size(labels, adjacency)
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(labels, adjacency)
            
            if new_cut > old_cut:
                improved = True
            else:
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
    
    return labels

def _spectral_embedding(adjacency):
    """
    Compute a spectral embedding of the graph for use as a geometric
    starting point. We use the Laplacian eigenvectors corresponding to
    the smallest non-trivial eigenvalues.
    """
    n = adjacency.shape[0]
    # Build the Laplacian: L = D - A
    degree = adjacency.sum(axis=1)
    D = np.diag(degree)
    L = D - adjacency.astype(float)
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to random if eigen decomposition fails
        return np.random.randn(n, 2)
    
    # Sort by eigenvalue (eigh already returns in ascending order)
    # The first eigenvector is the constant vector (eigenvalue 0)
    # We take the next few eigenvectors for the embedding
    # Use up to min(n, 10) eigenvectors for the embedding
    num_components = min(n, 10)
    embedding = eigenvectors[:, 1:num_components+1]
    
    return embedding

def _random_hyperplane_partition(embedding, deadline):
    """
    Given a spectral embedding, generate a random hyperplane and
    partition vertices based on which side of the hyperplane they fall on.
    """
    n, d = embedding.shape
    
    # Generate a random hyperplane normal vector
    normal = np.random.randn(d)
    
    # Project vertices onto the normal
    projections = embedding @ normal
    
    # Partition: vertices with positive projection get label 1, negative get 0
    labels = (projections > 0).astype(int)
    
    return labels

def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Compute spectral embedding once
    embedding = _spectral_embedding(adjacency)
    
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 4 == 0:
            # Random hyperplane over spectral embedding
            labels = _random_hyperplane_partition(embedding, deadline)
        elif restart % 4 == 1:
            # Alternating (from inspiration)
            labels = np.arange(n) % 2
        elif restart % 4 == 2:
            # Random with slight bias
            labels = np.random.randint(0, 2, n)
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        else:
            # Start from alternating but with random perturbation
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.2
            labels[flip_mask] = 1 - labels[flip_mask]
        
        # Local improvement
        labels = _local_improvement(labels.copy(), adjacency, deadline)
        
        # Second-order improvement
        labels = _second_order_improvement(labels, adjacency, deadline)
        
        # Compute cut size
        cut = _cut_size(labels, adjacency)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final check: if we never found a valid result, use a simple default
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
