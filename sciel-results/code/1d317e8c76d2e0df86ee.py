"""Maximum Cut solver using Goemans-Williamson SDP rounding + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    return int(adj[labels[:, None] != labels[None, :]].sum() // 2)


def _compute_gains(adj_int, labels):
    """
    Compute for each vertex the gain of flipping it.
    gain[v] = (# neighbors with opposite label) - (# neighbors with same label)
    """
    n = len(labels)
    degree = adj_int.sum(axis=1)
    
    # same_deg[v] = sum_u adj_int[v, u] * (labels[u] == labels[v])
    M = (labels[None, :] == labels[:, None])
    same_deg = (adj_int * M).sum(axis=1)
    
    gains = degree - 2 * same_deg
    return gains


def _local_search(adj, labels, max_passes=20):
    """
    Greedy local search: repeatedly flip vertices with positive gain.
    """
    n = len(labels)
    adj_int = adj.astype(np.int64)
    labels = labels.copy()
    
    for _ in range(max_passes):
        gains = _compute_gains(adj_int, labels)
        # Find all vertices with positive gain
        to_flip = np.where(gains > 0)[0]
        if len(to_flip) == 0:
            break
        # Flip all of them simultaneously (they are independent in the sense
        # that we're doing a batch update; this is a heuristic)
        for v in to_flip:
            labels[v] = 1 - labels[v]
    
    return labels


def _spectral_init(adj, seed=None):
    """
    Spectral initialization using the Fiedler vector (2nd eigenvector of Laplacian).
    This leverages graph structure to find a high-quality starting partition.
    """
    n = len(adj)
    adj = np.array(adj, dtype=float)
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except Exception:
        return rng.randint(0, 2, n)
    
    # Fiedler vector is the 2nd smallest eigenvector (index 1)
    fiedler = eigenvectors[:, 1]
    
    # Round by sign
    labels = (fiedler > 0).astype(int)
    
    # Ensure both classes are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = rng.randint(0, 2, n)
    
    return labels


def _goemans_williamson(adj, seed=None):
    """
    Goemans-Williamson SDP rounding for Max Cut.
    
    1. Solve the SDP relaxation (approximately via random projection for speed)
    2. Round using a random hyperplane
    
    For n=100, we can use a simpler approach: 
    - Use the eigenvectors of the Laplacian to get a good initial solution
    - Then use random projections in a high-dimensional space
    
    Actually, the standard GW algorithm:
    - The SDP solution gives vectors x_i in R^n such that ||x_i||=1 and x_i^T x_j >= 0
    - Then we pick a random vector r and set label_i = sign(x_i^T r)
    
    For our purposes, we can approximate the SDP solution using the fact that
    for Max Cut, the SDP optimum is related to the Laplacian eigenvalues.
    
    A practical approach: use multiple random projections of the Fiedler vector
    and other eigenvectors to generate diverse initial solutions.
    """
    n = len(adj)
    adj = np.array(adj, dtype=float)
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Get eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except Exception:
        # Fallback to random
        return rng.randint(0, 2, n)
    
    # Use the top few eigenvectors (excluding the trivial one)
    # The Fiedler vector (2nd smallest) is most important
    # But we can combine several for diversity
    
    # Strategy: pick a random linear combination of the top k eigenvectors
    # (excluding the first, which is the constant vector)
    k = min(5, n - 1)
    top_evecs = eigenvectors[:, 1:1+k]
    
    # Random weights
    weights = rng.randn(k)
    vec = top_evecs @ weights
    
    # Round by sign
    labels = (vec > 0).astype(int)
    
    # Ensure both classes are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        # Use median split
        median_val = np.median(vec)
        labels = (vec > median_val).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = rng.randint(0, 2, n)
    
    return labels


def _random_hyperplane_rounding(adj, seed=None):
    """
    Alternative initialization: random hyperplane rounding.
    Map vertices to random points on the unit sphere, then use a random
    hyperplane to partition.
    """
    n = len(adj)
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Random points on unit sphere in R^d
    d = 10
    points = rng.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / (norms + 1e-10)
    
    # Random hyperplane
    hyperplane = rng.randn(d)
    
    # Partition by sign of projection
    projections = points @ hyperplane
    labels = (projections > 0).astype(int)
    
    # Ensure both classes non-empty
    if labels.sum() == 0 or labels.sum() == n:
        labels = rng.randint(0, 2, n)
    
    return labels


def _simulated_annealing(adj, labels, max_iter=5000, t0=1.0, t1=0.01, seed=None):
    """
    Simulated annealing to escape local optima.
    Accepts worse solutions with probability exp(-delta/T).
    """
    n = len(labels)
    adj_int = adj.astype(np.int64)
    labels = labels.copy()
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    for i in range(max_iter):
        # Temperature schedule: exponential cooling
        T = t0 * (t1 / t0) ** (i / max_iter)
        
        # Pick a random vertex to flip
        v = rng.randint(n)
        old_label = labels[v]
        new_label = 1 - old_label
        
        # Compute delta: change in cut size
        # delta = (neighbors with opposite label after flip) - (neighbors with opposite label before flip)
        # Before flip: neighbors with label != old_label contribute to cut
        # After flip: neighbors with label == old_label contribute to cut (since they're now opposite)
        
        # Count neighbors with same label as v (old_label)
        same_count = adj_int[v, labels == old_label].sum()
        # Count neighbors with opposite label to v
        opp_count = adj_int[v, labels != old_label].sum()
        
        # Before flip: opp_count edges cross
        # After flip: same_count edges cross (those that were same are now opposite)
        # Delta = same_count - opp_count
        delta = same_count - opp_count
        
        if delta >= 0:
            # Accept (improvement or neutral)
            labels[v] = new_label
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(-delta/T)
            if T > 0 and rng.random() < np.exp(-delta / T):
                labels[v] = new_label
                current_cut += delta
    
    return best_labels


def _pairwise_swap_search(adj, labels):
    """
    Local search using pairwise swaps: swap labels of two vertices.
    This can escape local optima that single-flip cannot.
    """
    n = len(labels)
    adj_int = adj.astype(np.int64)
    labels = labels.copy()
    
    current_cut = _cut_size(adj, labels)
    
    for _ in range(5):  # A few passes
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                # Try swapping i and j
                new_labels = labels.copy()
                new_labels[i], new_labels[j] = labels[j], labels[i]
                new_cut = _cut_size(adj, new_labels)
                if new_cut > current_cut:
                    labels = new_labels
                    current_cut = new_cut
                    improved = True
        if not improved:
            break
    
    return labels


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=bool)
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    def try_solution(labels):
        nonlocal best_labels, best_cut
        # Local search
        improved = _local_search(adj, labels, max_passes=15)
        cut = _cut_size(adj, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
        return cut
    
    # Phase 1: Spectral initialization (Fiedler vector) - high-quality structural start
    if time.time() - start_time < deadline:
        labels = _spectral_init(adj, seed=42)
        try_solution(labels)
    
    # Phase 2: Goemans-Williamson with multiple random seeds
    for seed in range(30):
        if time.time() - start_time > deadline:
            break
        labels = _goemans_williamson(adj, seed=seed)
        try_solution(labels)
    
    # Phase 3: Random hyperplane rounding
    for seed in range(20):
        if time.time() - start_time > deadline:
            break
        labels = _random_hyperplane_rounding(adj, seed=seed)
        try_solution(labels)
    
    # Phase 4: Random restarts
    for restart in range(30):
        if time.time() - start_time > deadline:
            break
        labels = np.random.randint(0, 2, n)
        try_solution(labels)
    
    # Phase 5: Simulated annealing from best solution to escape local optima
    if time.time() - start_time < deadline and best_labels is not None:
        annealed = _simulated_annealing(adj, best_labels, max_iter=8000, t0=1.5, t1=0.005, seed=123)
        cut = _cut_size(adj, annealed)
        if cut > best_cut:
            best_cut = cut
            best_labels = annealed.copy()
            # Follow up with local search on the annealed solution
            polished = _local_search(adj, best_labels, max_passes=10)
            cut = _cut_size(adj, polished)
            if cut > best_cut:
                best_cut = cut
                best_labels = polished
    
    # Phase 6: Pairwise swap search on best solution
    if time.time() - start_time < deadline and best_labels is not None:
        swapped = _pairwise_swap_search(adj, best_labels)
        cut = _cut_size(adj, swapped)
        if cut > best_cut:
            best_cut = cut
            best_labels = swapped
    
    # Phase 7: Final local search polish
    if time.time() - start_time < deadline and best_labels is not None:
        final = _local_search(adj, best_labels, max_passes=30)
        cut = _cut_size(adj, final)
        if cut > best_cut:
            best_cut = cut
            best_labels = final
    
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels = _local_search(adj, labels, max_passes=20)
    
    return best_labels
# EVOLVE-BLOCK-END
