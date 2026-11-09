"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # For each edge (i,j), it crosses if labels[i] != labels[j]
    # Since adjacency is symmetric, we count each edge twice, so divide by 2
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    # Degree matrix
    d = A.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(d) - A
    
    # Compute eigenvalues and eigenvectors
    # We want the second smallest eigenvalue's eigenvector (Fiedler vector)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to random
        return np.random.randint(0, 2, size=n)
    
    # Sort by eigenvalues
    idx = np.argsort(eigvals)
    # Fiedler vector is the eigenvector corresponding to second smallest eigenvalue
    fiedler = eigvecs[:, idx[1]]
    
    # Partition based on sign of Fiedler vector
    labels = (fiedler >= 0).astype(np.int32)
    
    # Ensure both partitions are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _local_search(adjacency, labels, deadline, max_rounds=200):
    """Iterative local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    
    # Precompute adjacency as int array for speed
    A = adjacency.astype(np.int32)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            
            lv = labels[v]
            # Count neighbors in same partition vs opposite partition
            same_mask = (labels == lv)
            opp_mask = (labels != lv)
            
            same_count = A[v, same_mask].sum()
            opp_count = A[v, opp_mask].sum()
            
            # If flipping improves the cut
            if same_count > opp_count:
                labels[v] = 1 - lv
                improved = True
        
        if not improved:
            break
    
    return labels


def _simulated_annealing(adjacency, labels, deadline, max_iterations=5000):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    T = T_start
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Update temperature
        progress = it / max_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        new_cut = _cut_size(adjacency, labels)
        delta = new_cut - current_cut
        
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label
    
    return best_labels, best_cut


def partition(adjacency):
    """Main partition function using spectral initialization, local search, and simulated annealing."""
    n = len(adjacency)
    
    # Set a deadline with safety margin (150s limit, use 130s to be safe)
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Multiple strategies: spectral, random, and alternating
    candidates = []
    
    # 1. Spectral partition
    spec_labels = _spectral_partition(adjacency)
    candidates.append(spec_labels)
    
    # 2. Random partitions (a few)
    for _ in range(3):
        rand_labels = _random_labels(n)
        candidates.append(rand_labels)
    
    # 3. Alternating partition
    alt_labels = np.arange(n) % 2
    candidates.append(alt_labels)
    
    # For each candidate, run local search
    best_labels = None
    best_cut = -1
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Run local search on this candidate
        ls_labels = _local_search(adjacency, cand, deadline)
        ls_cut = _cut_size(adjacency, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
    
    # If we have time, run simulated annealing from the best local search result
    if time.time() < deadline - 10.0 and best_labels is not None:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
        
        # After SA, do one more local search polish
        if time.time() < deadline - 5.0:
            final_labels = _local_search(adjacency, best_labels, deadline)
            final_cut = _cut_size(adjacency, final_labels)
            if final_cut > best_cut:
                best_cut = final_cut
                best_labels = final_labels
    
    # Ensure we return a valid label array
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
