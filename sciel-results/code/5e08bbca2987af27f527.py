"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    labels = (fiedler >= 0).astype(np.int32)
    
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _gwg_spectral_partition(adjacency, n_hyperplanes=30):
    """
    Generate high-quality initial partitions using Goemans-Williamson style
    random hyperplane cuts from Laplacian eigenvectors.
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    eigvecs_sorted = eigvecs[:, idx]
    
    k = min(10, n - 1)
    embedding = eigvecs_sorted[:, 1:k+1]
    
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding_normalized = embedding / norms
    
    best_labels = None
    best_cut = -1
    
    for _ in range(n_hyperplanes):
        normal = np.random.randn(k)
        normal /= (np.linalg.norm(normal) + 1e-10)
        projections = embedding_normalized @ normal
        labels = (projections >= 0).astype(np.int32)
        
        if labels.sum() == 0 or labels.sum() == n:
            continue
        
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    if best_labels is None:
        fiedler = eigvecs_sorted[:, 1]
        best_labels = (fiedler >= 0).astype(np.int32)
        if best_labels.sum() == 0 or best_labels.sum() == n:
            best_labels = np.random.randint(0, 2, size=n)
    
    return best_labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _vectorized_local_search(adjacency, labels, deadline, max_rounds=200):
    """
    Vectorized local search: compute all gains at once per round using matrix ops.
    Gain of flipping vertex v = (num same-partition neighbors) - (num opposite-partition neighbors)
    If gain > 0, flipping increases cut size.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    degrees = A.sum(axis=1).astype(np.int32)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute same-partition neighbor counts for all vertices simultaneously
        same_mask = (labels[:, None] == labels[None, :])
        same_counts = (A * same_mask).sum(axis=1)
        
        # gain[v] = 2*same_counts[v] - degrees[v]
        gains = 2 * same_counts - degrees
        
        # Find vertices where flipping improves
        flip_mask = gains > 0
        if not flip_mask.any():
            break
        
        # Flip all improving vertices simultaneously (parallel update)
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels


def _adaptive_simulated_annealing(adjacency, labels, deadline, max_iterations=20000):
    """
    Simulated annealing with adaptive cooling based on acceptance rate.
    Uses vectorized gain computation for speed.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    degrees = A.sum(axis=1).astype(np.int32)
    
    # Adaptive temperature parameters
    T = 10.0
    T_min = 0.01
    T_max = 20.0
    cooling_rate = 0.95
    heating_rate = 1.1
    
    # Track acceptance in a window
    window_size = 100
    accept_count = 0
    total_count = 0
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Adaptive temperature adjustment
        if total_count >= window_size:
            acceptance_rate = accept_count / total_count
            if acceptance_rate < 0.1:
                # Too few accepts - heat up slightly
                T = min(T * heating_rate, T_max)
            elif acceptance_rate > 0.5:
                # Too many accepts - cool faster
                T = max(T * (cooling_rate * 0.9), T_min)
            else:
                # Normal cooling
                T = max(T * cooling_rate, T_min)
            accept_count = 0
            total_count = 0
        
        # Pick a random vertex
        v = np.random.randint(n)
        old_label = labels[v]
        
        # Compute gain for flipping v
        # same_count = number of neighbors with same label as v
        # opp_count = degree - same_count
        # gain = same_count - opp_count = 2*same_count - degree[v]
        same_count = (A[v] * (labels == old_label)).sum()
        delta = 2 * same_count - degrees[v]
        
        total_count += 1
        
        if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - old_label
            current_cut += delta
            accept_count += 1
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """Main partition function using spectral initialization, vectorized local search, and multi-start adaptive SA."""
    n = len(adjacency)
    
    # Set a deadline with safety margin (150s limit, use 140s to be safe)
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Standard spectral partition (Fiedler vector)
    spec_labels = _spectral_partition(adjacency)
    candidates.append(spec_labels)
    
    # 2. Goemans-Williamson style spectral partition with random hyperplanes
    gw_labels = _gwg_spectral_partition(adjacency, n_hyperplanes=30)
    candidates.append(gw_labels)
    
    # 3. Random partitions (a few)
    for _ in range(3):
        rand_labels = _random_labels(n)
        candidates.append(rand_labels)
    
    # 4. Alternating partition
    alt_labels = np.arange(n) % 2
    candidates.append(alt_labels)
    
    # For each candidate, run vectorized local search
    best_labels = None
    best_cut = -1
    ls_results = []
    
    for cand in candidates:
        if time.time() > deadline - 20.0:
            break
        
        # Run local search on this candidate
        ls_labels = _vectorized_local_search(adjacency, cand, deadline)
        ls_cut = _cut_size(adjacency, ls_labels)
        
        ls_results.append((ls_labels, ls_cut))
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
    
    # Multi-start adaptive simulated annealing
    # Sort LS results by cut size (best first)
    ls_results.sort(key=lambda x: x[1], reverse=True)
    
    # Run SA from the top 3 LS results if we have time
    n_sa_starts = min(3, len(ls_results))
    
    for i in range(n_sa_starts):
        if time.time() > deadline - 5.0:
            break
        
        sa_start_labels, sa_start_cut = ls_results[i]
        
        # Give more time to the first (best) SA start
        remaining = deadline - time.time()
        if i == 0:
            sa_deadline = time.time() + remaining * 0.6
        elif i == 1:
            sa_deadline = time.time() + remaining * 0.25
        else:
            sa_deadline = time.time() + remaining * 0.15
        
        sa_labels, sa_cut = _adaptive_simulated_annealing(adjacency, sa_start_labels, sa_deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final local search polish on the best result
    if best_labels is not None and time.time() < deadline - 2.0:
        final_labels = _vectorized_local_search(adjacency, best_labels, deadline)
        final_cut = _cut_size(adjacency, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    # Ensure we return a valid label array
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
