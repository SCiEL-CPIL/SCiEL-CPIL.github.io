"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # cut = sum over v of (number of neighbors of v in opposite partition)
    # Using matrix: labels are 0/1, 1-labels is opposite
    # adj @ (1 - labels) gives for each v the count of neighbors in opposite set
    # But this counts each edge twice? No: adj[v,u] is 1 if edge exists.
    # sum_v (adj[v] @ (1-labels)) = sum_{v,u} adj[v,u] * (1-labels[u])
    # For an edge (v,u) with labels[v]=0, labels[u]=1: contributes 1 from v's term
    # For an edge (v,u) with labels[v]=1, labels[u]=0: contributes 1 from u's term
    # So each crossing edge is counted exactly once in the sum? Let's check:
    # If labels[v]=0, labels[u]=1: adj[v,u]*(1-labels[u]) = 1*0 = 0, adj[u,v]*(1-labels[v]) = 1*1 = 1
    # So yes, each crossing edge contributes exactly 1.
    return np.sum(adj @ (1 - labels))


def _spectral_init(adj):
    """Initialize labels using the Fiedler vector of the Laplacian."""
    n = len(adj)
    adj = np.array(adj, dtype=float)
    degree = adj.sum(axis=1)
    lap = np.diag(degree) - adj
    # Use eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(lap)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        # Split at median
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(float)
    except Exception:
        labels = np.random.rand(n) > 0.5
        labels = labels.astype(float)
    return labels


def _greedy_flip_improve(adj, labels, max_iters=1000):
    """Greedy local search: flip a vertex if it improves the cut."""
    n = len(adj)
    adj = np.array(adj, dtype=float)
    labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    
    for _ in range(max_iters):
        improved = False
        # For each vertex, compute the gain from flipping
        # gain[v] = 2 * (number of neighbors in opposite set) - degree[v]
        # Actually: if we flip v, the change in cut is:
        # New edges crossing from v: neighbors in same set become crossing
        # Old edges crossing from v: neighbors in opposite set stop crossing
        # gain = (same_count - opposite_count) where same/opposite are current
        # But we want to maximize, so flip if gain > 0, i.e., same > opposite
        
        for v in range(n):
            same = np.sum(adj[v] * (labels == labels[v]))
            opposite = np.sum(adj[v] * (labels != labels[v]))
            if same > opposite:
                labels[v] = 1 - labels[v]
                improved = True
                new_cut = _cut_size(adj, labels)
                if new_cut > best_cut:
                    best_cut = new_cut
        if not improved:
            break
    
    return labels, best_cut


def _simulated_annealing(adj, labels, start_temp=10.0, end_temp=0.01, n_iter=2000):
    """Simulated annealing to escape local optima."""
    n = len(adj)
    adj = np.array(adj, dtype=float)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for i in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        same = np.sum(adj[v] * (labels == labels[v]))
        opposite = np.sum(adj[v] * (labels != labels[v]))
        delta = opposite - same  # change in cut if we flip
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            labels[v] = 1 - labels[v]
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """Main partition function using spectral init + greedy + simulated annealing."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    n = len(adjacency)
    adj = np.array(adjacency, dtype=float)
    
    # Multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    # Start with spectral initialization
    labels_spec = _spectral_init(adj)
    labels, cut = _greedy_flip_improve(adj, labels_spec, max_iters=500)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels.astype(int)
    
    # Simulated annealing from the best solution
    labels_sa, cut_sa = _simulated_annealing(adj, best_labels, start_temp=5.0, end_temp=0.01, n_iter=1000)
    if cut_sa > best_cut:
        best_cut = cut_sa
        best_labels = labels_sa.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels.astype(int)
    
    # Greedy polish from SA result
    labels_polish, cut_polish = _greedy_flip_improve(adj, best_labels, max_iters=500)
    if cut_polish > best_cut:
        best_cut = cut_polish
        best_labels = labels_polish.copy()
    
    # Try a few more random restarts if time permits
    while time.time() < deadline - 5:
        # Random initialization
        labels_rand = (np.random.rand(n) > 0.5).astype(float)
        labels, cut = _greedy_flip_improve(adj, labels_rand, max_iters=200)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Short SA
        labels_sa, cut_sa = _simulated_annealing(adj, best_labels, start_temp=3.0, end_temp=0.01, n_iter=300)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
        
        # Polish
        labels_polish, cut_polish = _greedy_flip_improve(adj, best_labels, max_iters=200)
        if cut_polish > best_cut:
            best_cut = cut_polish
            best_labels = labels_polish.copy()
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
