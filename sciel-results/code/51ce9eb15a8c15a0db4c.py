"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Convert to numpy array if not already
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Build Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Convert to sparse matrix
    L_sparse = sp.csr_matrix(L)
    
    # Determine number of eigenvectors to extract
    # We want enough to capture the structure, but not too many to be slow
    # For n=100, k=20 is reasonable
    k = min(20, n - 1)
    
    try:
        # Find smallest eigenvalues
        # sigma=1e-10 helps with convergence for small eigenvalues
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, which='SM', sigma=1e-10, return_eigenvectors=True)
        
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Filter out trivial eigenvectors (eigenvalue close to 0)
        # This handles disconnected graphs where multiple eigenvalues are 0
        non_trivial_mask = eigvals > 1e-6
        if np.sum(non_trivial_mask) > 0:
            embedding = eigvecs[:, non_trivial_mask]
        else:
            # If all are trivial (e.g. empty graph or n=1 handled above), fallback
            embedding = eigvecs
            
    except Exception:
        # Fallback to random initialization if eigensolver fails
        labels = np.random.randint(0, 2, n)
        labels = local_improve(adj, labels, n, max_rounds=20)
        return labels.astype(int)
    
    # Ensure embedding has at least 1 dimension
    if embedding.shape[1] == 0:
        embedding = np.random.randn(n, 1)

    # Now perform randomized hyperplane projection
    best_labels = None
    best_cut = -1
    
    # Increase number of trials for better coverage
    n_hyperplanes = 50
    
    for trial in range(n_hyperplanes):
        # Generate a random unit vector in the embedding space
        dim = embedding.shape[1]
        random_vec = np.random.randn(dim)
        norm = np.linalg.norm(random_vec)
        if norm < 1e-10:
            continue
        random_vec /= norm
        
        # Project embedding onto this hyperplane
        projections = embedding @ random_vec
        
        # Partition: positive projections -> 1, negative -> 0
        labels = (projections >= 0).astype(int)
        
        # Ensure both partitions are non-empty (if possible)
        if labels.sum() == 0 or labels.sum() == n:
            # Flip a few random vertices to break symmetry
            flip_idx = np.random.choice(n, size=max(1, n // 10), replace=False)
            labels[flip_idx] = 1 - labels[flip_idx]
        
        # Local improvement with more rounds
        labels = local_improve(adj, labels, n, max_rounds=20)
        
        # Compute cut size
        cut_size = compute_cut_size(adj, labels)
        
        if cut_size > best_cut:
            best_cut = cut_size
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.random.randint(0, 2, n)
        best_labels = local_improve(adj, best_labels, n, max_rounds=20)
    
    return best_labels.astype(int)


def compute_cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Vectorized computation
    # labels: (n,), adj: (n, n)
    # cross[i, j] is 1 if labels[i] != labels[j]
    cross = (labels[:, None] != labels[None, :]).astype(np.float64)
    cut_size = (adj * cross).sum() / 2
    return cut_size


def local_improve(adj, labels, n, max_rounds=20):
    """Perform local search to improve the partition."""
    labels = labels.copy()
    
    for _ in range(max_rounds):
        improved = False
        # Randomize the order of vertices for diversity
        order = np.random.permutation(n)
        for v in order:
            # Compute gain of flipping v
            # current label: labels[v]
            # neighbors with same label: adj[v][labels == labels[v]]
            # neighbors with diff label: adj[v][labels != labels[v]]
            # If diff > same, flip
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if other > same:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
