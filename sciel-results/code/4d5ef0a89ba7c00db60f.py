"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # Count edges where endpoints have different labels
    # For undirected graph, each edge is counted twice in the matrix, so divide by 2
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_init(adjacency, n, rng):
    """Initialize labels using spectral method (Fiedler vector of Laplacian)."""
    # Build Laplacian L = D - A
    degree = adjacency.sum(axis=1)
    # Use eigenvalue decomposition of the Laplacian
    # For efficiency, we can use the fact that we want the second eigenvector
    # L = D - A
    L = np.diag(degree) - adjacency
    
    # Use numpy's eigvalsh for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
    except:
        # Fallback to random
        return (rng.random(n) > 0.5).astype(np.int32)
    
    # Assign labels based on sign of Fiedler vector
    labels = (fiedler >= 0).astype(np.int32)
    # Ensure both labels are present
    if labels.sum() == 0 or labels.sum() == n:
        # Flip half randomly
        flip_idx = rng.choice(n, size=n // 2, replace=False)
        labels[flip_idx] = 1 - labels[flip_idx]
    return labels


def _random_init(n, rng):
    """Random initialization."""
    return (rng.random(n) > 0.5).astype(np.int32)


def _local_search(adjacency, labels, deadline, max_rounds=200):
    """Vectorized local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute for each vertex: how many edges go to same side vs opposite side
        # same[v] = number of neighbors of v with same label as v
        # opp[v] = number of neighbors of v with opposite label to v
        # If same[v] > opp[v], flipping v increases the cut
        
        same_mask = (labels[:, None] == labels[None, :])
        opp_mask = ~same_mask
        
        same_counts = (adjacency & same_mask).sum(axis=1)
        opp_counts = (adjacency & opp_mask).sum(axis=1)
        
        # Gain from flipping vertex v = opp[v] - same[v]
        # If gain > 0, flipping improves the cut
        gains = opp_counts - same_counts
        
        # Find vertices that would improve if flipped
        to_flip = gains > 0
        
        if not to_flip.any():
            break
        
        labels[to_flip] = 1 - labels[to_flip]
    
    return labels


def partition(adjacency):
    """Find a good 2-partition maximizing the number of crossing edges."""
    adjacency = np.asarray(adjacency, dtype=bool)
    n = len(adjacency)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # Try multiple initializations
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Alternate between spectral and random initialization
        if restart == 0:
            labels = _spectral_init(adjacency, n, rng)
        elif restart % 2 == 1:
            labels = _random_init(n, rng)
        else:
            # Perturbed spectral: flip some random vertices from spectral init
            labels = _spectral_init(adjacency, n, rng)
            if n > 10:
                flip_count = max(1, n // 10)
                flip_idx = rng.choice(n, size=flip_count, replace=False)
                labels[flip_idx] = 1 - labels[flip_idx]
        
        # Local search
        labels = _local_search(adjacency, labels, deadline)
        
        # Evaluate
        cut = _cut_size(adjacency, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
