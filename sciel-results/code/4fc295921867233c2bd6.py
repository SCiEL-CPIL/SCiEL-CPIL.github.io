"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    return int(adj[labels[:, None] != labels[None, :]].sum() // 2)


def _compute_gains(adj_int, labels):
    """
    Compute for each vertex the gain (positive) or loss (negative) of flipping it.
    gain[v] = (# neighbors with opposite label) - (# neighbors with same label)
    If gain[v] > 0, flipping v increases the cut.
    """
    n = len(labels)
    # For vertex v:
    same_count = np.zeros(n, dtype=np.int64)
    other_count = np.zeros(n, dtype=np.int64)
    
    # Using matrix operations:
    # labels is 0/1. For each vertex v, count neighbors with same label and different label.
    # adj_int is n x n.
    # same_mask[v, u] = adj_int[v, u] and labels[u] == labels[v]
    
    # Efficient approach:
    # For each vertex v, sum over neighbors u of adj_int[v,u] where labels[u] == labels[v]
    # and where labels[u] != labels[v]
    
    # Let's compute degree and same-degree:
    # degree[v] = sum_u adj_int[v, u]
    # same_deg[v] = sum_u adj_int[v, u] * (labels[u] == labels[v])
    # other_deg[v] = degree[v] - same_deg[v]
    # gain[v] = other_deg[v] - same_deg[v] = degree[v] - 2 * same_deg[v]
    
    degree = adj_int.sum(axis=1)
    
    # same_deg[v] = sum_u adj_int[v, u] * (labels[u] == labels[v])
    # We can compute this as: for each label class, compute adjacency sum
    # Let's use: same_deg = adj_int @ (labels == labels[None, :]).T ... that's not right.
    
    # Better: create a matrix M where M[v, u] = 1 if labels[u] == labels[v] else 0
    # Then same_deg[v] = sum_u adj_int[v, u] * M[v, u]
    
    # M[v, u] = (labels[u] == labels[v])
    # This is an outer product-like thing.
    # M = (labels[None, :] == labels[:, None])  # M[v, u] = (labels[u] == labels[v])
    # Then same_deg = (adj_int * M).sum(axis=1)
    
    M = (labels[None, :] == labels[:, None])
    same_deg = (adj_int * M).sum(axis=1)
    
    gains = degree - 2 * same_deg
    return gains


def _greedy_improve_fast(adj, labels):
    """
    Fast greedy local search using incremental gain updates.
    Returns improved labels and the cut size.
    """
    n = len(labels)
    adj_int = adj.astype(np.int64)
    labels = labels.copy()
    
    # Compute initial gains
    gains = _compute_gains(adj_int, labels)
    
    improved = True
    max_iterations = n * 10  # Safety bound
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for v in range(n):
            if gains[v] > 0:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                
                # Update gains for v and its neighbors
                # When we flip v, the gain for v changes sign (roughly)
                # Actually, let's recompute gains for affected vertices.
                # Flipping v affects:
                # 1. The gain of v itself
                # 2. The gain of each neighbor u of v
                
                # For efficiency, we can update incrementally.
                # But for simplicity and correctness, let's recompute gains for v and its neighbors.
                
                # Actually, a simpler approach: recompute all gains. For n=100, this is fast.
                gains = _compute_gains(adj_int, labels)
                improved = True
    
    cut = _cut_size(adj, labels)
    return labels, cut


def _spectral_init(adj):
    """
    Initialize labels using the second eigenvector of the graph Laplacian.
    The Fiedler vector gives a good initial partition.
    """
    n = len(adj)
    adj = np.array(adj, dtype=float)
    
    # Compute Laplacian: L = D - A
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Find the second smallest eigenvalue and its eigenvector
    # Use numpy's eigvalsh for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvalue is at index 1 (index 0 is the smallest, which is 0)
        fiedler = eigenvectors[:, 1]
        
        # Use the sign of the Fiedler vector to create initial labels
        # Vertices with positive Fiedler value get label 1, negative get label 0
        labels = (fiedler > 0).astype(int)
        
        # Handle the case where all values are on one side (shouldn't happen for connected graphs)
        if labels.sum() == 0 or labels.sum() == n:
            # Fallback: split by median
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(int)
        
    except Exception:
        # Fallback to alternating
        labels = np.arange(n) % 2
    
    return labels


def _simulated_annealing(adj, labels, start_time, deadline):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    adj_int = adj.astype(np.int64)
    labels = labels.copy()
    
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Temperature schedule
    T = 10.0
    cooling = 0.95
    
    while time.time() - start_time < deadline:
        if T < 0.01:
            break
        
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        new_labels = labels.copy()
        new_labels[v] = 1 - new_labels[v]
        
        new_cut = _cut_size(adj, new_labels)
        delta = new_cut - current_cut
        
        if delta > 0:
            # Accept improvement
            labels = new_labels
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(delta / T)
            if random.random() < np.exp(delta / T):
                labels = new_labels
                current_cut = new_cut
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=bool)
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + greedy improvement
    spectral_labels = _spectral_init(adj)
    improved_labels, cut = _greedy_improve_fast(adj, spectral_labels)
    
    if cut > best_cut:
        best_cut = cut
        best_labels = improved_labels.copy()
    
    # Phase 2: Multiple random restarts with greedy
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random initial labeling
        labels = np.random.randint(0, 2, n)
        
        # Greedy local improvement
        improved_labels, cut = _greedy_improve_fast(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved_labels.copy()
    
    # Phase 3: Simulated annealing from the best solution found
    if time.time() - start_time < deadline:
        sa_labels, sa_cut = _simulated_annealing(adj, best_labels, start_time, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # If no restarts completed (shouldn't happen), use alternating start
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels, _ = _greedy_improve_fast(adj, labels)
    
    return best_labels
# EVOLVE-BLOCK-END
