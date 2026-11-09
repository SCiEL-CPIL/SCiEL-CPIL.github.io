"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # 1. Compute Spectral Embedding (Proxy for SDP Solution Space)
    # The Goemans-Williamson SDP for Max-Cut involves variables v_i in R^k.
    # A common heuristic is to use the eigenvectors of the Laplacian or Adjacency matrix.
    # We use the Laplacian L = D - A. The smallest non-zero eigenvectors capture the community structure.
    
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Convert to sparse for efficiency
    L_sparse = sp.csr_matrix(L)
    
    # We want the eigenvectors corresponding to the smallest eigenvalues.
    # The first eigenvector is the constant vector (eigenvalue 0).
    # We take k smallest.
    k = min(10, n - 1)
    
    try:
        # Find k smallest eigenvalues
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, which='SM', return_eigenvectors=True)
        
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Filter out the trivial eigenvector (eigenvalue close to 0)
        # The trivial eigenvector is constant and doesn't help with partitioning.
        non_trivial_mask = eigvals > 1e-8
        if np.sum(non_trivial_mask) > 0:
            embedding = eigvecs[:, non_trivial_mask]
        else:
            # If graph is disconnected or something weird, use all
            embedding = eigvecs
            
    except Exception:
        # Fallback: Random embedding
        embedding = np.random.randn(n, 10)
        
    # 2. Geometric Processing of Embedding (SDP Constraints)
    # In the SDP, v_i are unit vectors. Let's normalize our embedding vectors to unit length.
    # Also, center the embedding to remove any bias from the mean.
    
    # Center
    mean_vec = np.mean(embedding, axis=0)
    embedding = embedding - mean_vec
    
    # Normalize rows to unit length (mimicking v_i^T v_i = 1)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding = embedding / norms
    
    # 3. Randomized Hyperplane Projection
    # Generate multiple random hyperplanes to project the embedding onto.
    # Each projection gives a 1D sign pattern, which is a partition.
    
    best_labels = None
    best_cut = -1
    
    n_hyperplanes = 50 # Increase trials for better exploration
    
    dim = embedding.shape[1]
    
    for trial in range(n_hyperplanes):
        # Generate random unit vector
        random_vec = np.random.randn(dim)
        random_vec /= np.linalg.norm(random_vec)
        
        # Project
        projections = embedding @ random_vec
        
        # Partition based on sign
        labels = (projections >= 0).astype(int)
        
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            # Flip random vertices to break symmetry
            flip_idx = np.random.choice(n, size=max(1, n // 10), replace=False)
            labels[flip_idx] = 1 - labels[flip_idx]
        
        # 4. Local Search Refinement
        labels = local_improve(adj, labels, n, max_rounds=20)
        
        # Compute Cut Size
        cut_size = compute_cut_size(adj, labels)
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_labels = local_improve(adj, best_labels, n, max_rounds=20)
    
    return best_labels.astype(int)


def compute_cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Efficient calculation:
    # Cut size = sum_{i,j} A_{ij} * I(labels[i] != labels[j])
    # Since A is symmetric, we can just sum all and divide by 2, or use matrix ops.
    # labels is 0/1. 
    # Let S be the set of nodes with label 1.
    # Cut = sum_{i in S, j not in S} A_{ij}
    
    mask = (labels == 1)
    # Sum of edges between mask and ~mask
    # adj[mask, ~mask].sum()
    cut = adj[mask, ~mask].sum()
    return cut


def local_improve(adj, labels, n, max_rounds=20):
    """Perform local search to improve the partition."""
    labels = labels.copy()
    
    for _ in range(max_rounds):
        improved = False
        # Randomize order for diversity
        order = np.random.permutation(n)
        for v in order:
            # Calculate gain of flipping v
            # Current label: labels[v]
            # Neighbors with same label: same_count
            # Neighbors with different label: diff_count
            # If we flip, the edges to 'same' become 'diff' (gain +same_count)
            # The edges to 'diff' become 'same' (loss -diff_count)
            # Net gain = same_count - diff_count
            # If gain > 0, flip.
            
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
