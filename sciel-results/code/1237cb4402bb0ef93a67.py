"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Use spectral method to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    # Compute eigenvector of A corresponding to smallest eigenvalue of A - 2D + 2I
    # Or simpler: use the Fiedler vector approach
    # For max cut, we want to maximize x^T A x subject to x in {-1,1}^n
    # Spectral relaxation: use the eigenvector of A with largest magnitude eigenvalue
    
    # Use numpy's eigh for symmetric matrix
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        # Use the eigenvector corresponding to the most negative eigenvalue
        # (this gives the best spectral partition for max cut)
        idx = np.argmin(eigvals)
        v = eigvecs[:, idx]
        # Partition based on sign of eigenvector
        labels = (v > 0).astype(np.int32)
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2


def _local_search(adjacency, labels, deadline, max_rounds=50):
    """Iterative local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Compute flip gains for all vertices
        same_mask = (labels[:, None] == labels[None, :])
        opp_mask = ~same_mask
        
        same_counts = (adjacency & same_mask).sum(axis=1)
        opp_counts = (adjacency & opp_mask).sum(axis=1)
        
        gains = same_counts - opp_counts
        
        # Find vertices where flipping improves the cut
        flip_candidates = np.where(gains > 0)[0]
        
        if len(flip_candidates) == 0:
            break
        
        # Flip all improving vertices at once (parallel update)
        labels[flip_candidates] = 1 - labels[flip_candidates]
        improved = True
    
    return labels


def _simulated_annealing(adjacency, labels, deadline, max_iters=5000):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        new_labels = labels.copy()
        new_labels[v] = 1 - new_labels[v]
        
        new_cut = _cut_size(adjacency, new_labels)
        delta = new_cut - current_cut
        
        if delta > 0:
            # Accept improving move
            labels = new_labels
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept non-improving move with probability exp(delta/T)
            if T > 0 and np.random.random() < np.exp(delta / T):
                labels = new_labels
                current_cut = new_cut
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 100  # 100s deadline with 20s safety margin
    
    # Get spectral initialization
    labels = _spectral_partition(adjacency)
    
    # Local search to polish
    labels = _local_search(adjacency, labels, deadline)
    best_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    
    # Simulated annealing with restarts to escape local optima
    num_restarts = 3
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start from best found or random perturbation
        if r == 0:
            sa_labels = best_labels.copy()
        else:
            sa_labels = best_labels.copy()
            # Perturb some vertices
            num_perturb = max(1, n // 10)
            perturb_indices = np.random.choice(n, num_perturb, replace=False)
            sa_labels[perturb_indices] = 1 - sa_labels[perturb_indices]
        
        sa_result, sa_cut = _simulated_annealing(adjacency, sa_labels, deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_result.copy()
    
    # Final local search polish
    if time.time() < deadline:
        best_labels = _local_search(adjacency, best_labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
