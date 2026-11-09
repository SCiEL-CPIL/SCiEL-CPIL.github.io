# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    # Current contribution of v to cut: number of neighbors with opposite label
    # If we flip v, its contribution becomes: number of neighbors with same label (old)
    # Gain = new_contribution - old_contribution
    opp = (labels != labels[v]).sum()
    same = (labels == labels[v]).sum()
    # Actually need to count only neighbors
    neighbor_labels = labels[adjacency[v]]
    opp_count = (neighbor_labels != labels[v]).sum()
    same_count = (neighbor_labels == labels[v]).sum()
    return same_count - opp_count


def _spectral_init(adjacency):
    """Use spectral method (eigenvectors of Laplacian) for initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    degrees = A.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(degrees) - A
    
    # Use scipy if available for eigenvalue computation
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import eigsh
        
        L_sparse = csc_matrix(L)
        # Get the smallest eigenvalues and eigenvectors
        # We want the Fiedler vector (2nd smallest eigenvalue)
        if n > 2:
            k = min(4, n - 1)
            vals, vecs = eigsh(L_sparse, k=k, which='SM')
            # Use the Fiedler vector (index 1, since index 0 is the trivial one)
            fiedler = vecs[:, 1]
        else:
            fiedler = np.random.randn(n)
    except Exception:
        # Fallback: simple random or alternating
        fiedler = np.random.randn(n)
    
    # Partition based on sign of Fiedler vector
    labels = (fiedler >= 0).astype(np.int8)
    return labels


def _walksat(adjacency, labels, deadline, max_iters=20000):
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    
    # Precompute total degrees
    total_degrees = adjacency.sum(axis=1)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Calculate gains for all vertices
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        best_gain = np.max(gains)
        
        if best_gain > 0:
            # Flip the vertex with the best gain
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            cut += best_gain
        else:
            # Local optimum. Pick a vertex involved in an uncut edge.
            uncut_mask = adjacency & (~diff)
            uncut_edges = np.argwhere(uncut_mask)
            
            if len(uncut_edges) == 0:
                break
                
            # Pick a random uncut edge
            edge_idx = np.random.randint(0, len(uncut_edges))
            u, v = uncut_edges[edge_idx]
            
            # Flip one of them randomly
            if np.random.rand() < 0.5:
                flip_v = u
            else:
                flip_v = v
                
            # Update cut
            cut += gains[flip_v]
            labels[flip_v] = 1 - labels[flip_v]
            
    return labels, cut


def _simulated_annealing(adjacency, labels, deadline, max_iters=50000):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = cut
    
    total_degrees = adjacency.sum(axis=1)
    current_temp = 5.0
    cooling_rate = 0.9995
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate gains for all vertices
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        # Choose a random vertex to flip
        v = np.random.randint(n)
        gain = gains[v]
        
        if gain > 0 or np.random.rand() < np.exp(gain / max(current_temp, 1e-8)):
            labels[v] = 1 - labels[v]
            cut += gain
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        current_temp *= cooling_rate
    
    return best_labels, best_cut


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin: 145s out of 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Use spectral init + WalkSAT + Simulated Annealing
    # Multiple restarts with different initializations
    
    num_restarts = 15
    base_seed = 42
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Spectral initialization
            labels = _spectral_init(adjacency)
        elif restart == 1:
            # Alternating start
            labels = np.arange(n) % 2
        elif restart % 3 == 2:
            # Spectral with random perturbation
            labels = _spectral_init(adjacency)
            # Perturb some vertices
            perturb_idx = np.random.choice(n, size=n//10, replace=False)
            labels[perturb_idx] = 1 - labels[perturb_idx]
        else:
            # Random start
            labels = _random_labels(n, base_seed + restart)
        
        # Run WalkSAT first
        labels, cut = _walksat(adjacency, labels, deadline)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we have time, run simulated annealing on the best result
        if time.time() < deadline - 5:
            sa_labels, sa_cut = _simulated_annealing(adjacency, labels, deadline)
            
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        # If we still have time, do another WalkSAT from the SA result
        if sa_cut > best_cut and time.time() < deadline - 2:
            ws_labels, ws_cut = _walksat(adjacency, sa_labels, deadline)
            if ws_cut > best_cut:
                best_cut = ws_cut
                best_labels = ws_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
