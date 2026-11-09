"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, max_passes=50):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    labels = labels.copy()
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            if labels[v] == 0:
                same_set = (labels == 0)
                same_set[v] = False
            else:
                same_set = (labels == 1)
                same_set[v] = False
            
            edges_to_same = A[v, same_set].sum()
            gain = 2 * edges_to_same - row_sums[v]
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels


def _spectral_partition(adjacency):
    """Use spectral method to get initial partition via Fiedler vector."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A  # Laplacian
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler >= 0).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, n)
        return labels
    except Exception:
        return np.random.randint(0, 2, n)


def _goemans_williamson_partition(adjacency, n_samples=50):
    """
    Goemans-Williamson style: use eigenvectors of Laplacian to define random hyperplane cuts.
    For each sample, pick a random unit vector in the subspace spanned by the first few
    eigenvectors (excluding the trivial constant one), and cut by sign of projection.
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A  # Laplacian
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Use eigenvectors corresponding to smallest non-zero eigenvalues
        # Skip the first (constant) eigenvector
        k = min(5, n - 1)  # Use up to 5 eigenvectors
        V = eigvecs[:, 1:1+k]  # Shape (n, k)
        
        best_labels = None
        best_cut = -1
        
        for _ in range(n_samples):
            # Random unit vector in R^k
            r = np.random.randn(k)
            r /= np.linalg.norm(r) + 1e-12
            # Project vertices onto this direction
            scores = V @ r  # Shape (n,)
            labels = (scores >= 0).astype(int)
            
            # Check if degenerate
            if labels.sum() == 0 or labels.sum() == n:
                continue
            
            cut = _cut_size(adjacency, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        if best_labels is None:
            return np.random.randint(0, 2, n)
        return best_labels
    except Exception:
        return np.random.randint(0, 2, n)


def _adaptive_sa(adjacency, labels, t0, t_min, deadline, max_iter=10000):
    """
    Simulated annealing with:
      - vectorized gain computation
      - adaptive cooling based on acceptance ratio
      - 1-flip and 2-flip moves to escape deeper local minima
    Returns (best_labels, best_cut).
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    
    current = labels.copy()
    current_cut = _cut_size(adjacency, current)
    best = current.copy()
    best_cut = current_cut
    
    T = t0
    # Track acceptance ratio for adaptive cooling
    accept_count = 0
    total_count = 0
    window_size = 100
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive cooling: adjust T based on recent acceptance rate
        if total_count >= window_size:
            ratio = accept_count / total_count
            # If too many rejections, cool faster; if too many acceptances, warm slightly
            if ratio < 0.05:
                T *= 0.85
            elif ratio > 0.6:
                T *= 1.1
            else:
                T *= 0.98
            T = max(T, t_min)
            accept_count = 0
            total_count = 0
        
        # If stuck for too long, increase temperature to escape local minimum
        if no_improve_count > max_no_improve:
            T = max(T * 1.5, t0 * 0.5)
            no_improve_count = 0
        
        # Choose move type: 1-flip (70%) or 2-flip (30%)
        use_two_flip = (np.random.random() < 0.3)
        
        if not use_two_flip:
            # 1-flip move
            v = np.random.randint(n)
            
            # Compute gain for flipping v
            # gain = 2 * (edges from v to same set) - row_sums[v]
            if current[v] == 0:
                same_mask = (current == 0)
            else:
                same_mask = (current == 1)
            same_mask[v] = False
            edges_to_same = A[v, same_mask].sum()
            delta = 2 * edges_to_same - row_sums[v]
            
            new_labels = current.copy()
            new_labels[v] = 1 - new_labels[v]
        else:
            # 2-flip move: pick two distinct vertices
            v1, v2 = np.random.choice(n, size=2, replace=False)
            
            # Compute combined gain for flipping both v1 and v2
            new_labels = current.copy()
            new_labels[v1] = 1 - current[v1]
            new_labels[v2] = 1 - current[v2]
            new_cut = _cut_size(adjacency, new_labels)
            delta = new_cut - current_cut
        
        total_count += 1
        
        # Accept/reject
        if delta > 0:
            current = new_labels
            current_cut = current_cut + delta
            if current_cut > best_cut:
                best_cut = current_cut
                best = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            accept_count += 1
        else:
            no_improve_count += 1
            if T > t_min:
                # Boltzmann acceptance
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current = new_labels
                    current_cut = current_cut + delta
                    accept_count += 1
    
    return best, best_cut


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the number of cut edges.
    Uses Goemans-Williamson spectral initialization, local search, and adaptive Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = len(adjacency)
    A = np.array(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Goemans-Williamson initialization + local search
    if time.time() < deadline - 20:
        labels = _goemans_williamson_partition(A, n_samples=80)
        labels = _local_search(A, labels)
        cut = _cut_size(A, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 2: Spectral initialization + local search
    if time.time() < deadline - 20:
        labels = _spectral_partition(A)
        labels = _local_search(A, labels)
        cut = _cut_size(A, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 3: Multiple random initializations + local search
    for _ in range(3):
        if time.time() < deadline - 20:
            labels = np.random.randint(0, 2, n)
            labels = _local_search(A, labels)
            cut = _cut_size(A, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    # Phase 4: Adaptive SA from best current state
    if best_labels is not None and time.time() < deadline - 15:
        sa_labels, sa_cut = _adaptive_sa(
            A, best_labels, t0=5.0, t_min=0.01,
            deadline=deadline - 10, max_iter=20000
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish with local search
        if time.time() < deadline - 5:
            best_labels = _local_search(A, best_labels)
            best_cut = _cut_size(A, best_labels)
    
    # Phase 5: Multiple random restarts with adaptive SA
    restarts = 0
    while time.time() < deadline - 15 and restarts < 10:
        # Use different initialization strategies
        if restarts % 4 == 0:
            labels = np.random.randint(0, 2, n)
        elif restarts % 4 == 1:
            # Start from best and perturb
            labels = best_labels.copy()
            num_perturb = max(1, n // 10)
            perturb_idx = np.random.choice(n, size=num_perturb, replace=False)
            labels[perturb_idx] = 1 - labels[perturb_idx]
        elif restarts % 4 == 2:
            # Start from Goemans-Williamson and perturb
            labels = _goemans_williamson_partition(A, n_samples=30)
            num_perturb = max(1, n // 10)
            perturb_idx = np.random.choice(n, size=num_perturb, replace=False)
            labels[perturb_idx] = 1 - labels[perturb_idx]
        else:
            # Start from spectral and perturb
            labels = _spectral_partition(A)
            num_perturb = max(1, n // 10)
            perturb_idx = np.random.choice(n, size=num_perturb, replace=False)
            labels[perturb_idx] = 1 - labels[perturb_idx]
        
        labels = _local_search(A, labels)
        
        # Run SA
        sa_labels, sa_cut = _adaptive_sa(
            A, labels, t0=5.0, t_min=0.01,
            deadline=deadline - 10, max_iter=10000
        )
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        restarts += 1
    
    # Final local search polish
    if best_labels is not None and time.time() < deadline - 2:
        best_labels = _local_search(A, best_labels, max_passes=20)
        final_cut = _cut_size(A, best_labels)
        if final_cut > best_cut:
            best_cut = final_cut
    
    return best_labels
# EVOLVE-BLOCK-END
