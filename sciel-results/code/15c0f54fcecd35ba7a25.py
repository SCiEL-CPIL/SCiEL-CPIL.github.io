"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    # For each vertex v, count neighbors with different label
    # cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)


def _flip_gain(adj, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    # Current contribution of v to cut: sum of adj[v,u] where labels[u] != labels[v]
    # After flip: sum of adj[v,u] where labels[u] == labels[v] (now different)
    # Gain = (new_crossing - old_crossing)
    # old_crossing = sum_u adj[v,u] * (labels[u] != labels[v])
    # new_crossing = sum_u adj[v,u] * (labels[u] == labels[v])
    # gain = new_crossing - old_crossing
    same = np.sum(adj[v] * (labels == labels[v]))
    other = np.sum(adj[v] * (labels != labels[v]))
    return int(other - same)  # Wait, let me reconsider


def _local_search(adj, labels, max_iters=100):
    """Greedy local search: flip vertices that improve the cut."""
    n = len(labels)
    adj = np.asarray(adj, dtype=np.int8)
    labels = labels.copy()
    
    for _ in range(max_iters):
        improved = False
        for v in range(n):
            # Count neighbors with same label vs different label
            same_count = np.sum(adj[v] * (labels == labels[v]))
            diff_count = np.sum(adj[v] * (labels != labels[v]))
            # If we flip v, the contribution changes from diff_count to same_count
            # Gain = same_count - diff_count
            if same_count > diff_count:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    return labels


def _simulated_annealing(adj, labels, start_temp=1.0, end_temp=0.01, cooling=0.995, max_iters=500):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    adj = np.asarray(adj, dtype=np.int8)
    labels = labels.copy()
    
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    temp = start_temp
    for it in range(max_iters):
        if temp <= end_temp:
            break
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        old_label = labels[v]
        labels[v] = 1 - old_label
        new_cut = _cut_size(adj, labels)
        delta = new_cut - current_cut
        
        if delta >= 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label  # revert
        
        temp *= cooling
    
    return best_labels


def _spectral_init(adj, n):
    """Initialize using Fiedler vector (second eigenvector of Laplacian)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    # Build Laplacian L = D - A
    degrees = np.sum(adj, axis=1)
    L = np.diag(degrees) - adj
    
    # Find Fiedler vector (smallest non-zero eigenvalue)
    try:
        # For symmetric matrix, use eigsh
        vals, vecs = eigsh(L, k=2, which='SM')
        # Fiedler vector is the eigenvector corresponding to second smallest eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    except Exception:
        # Fallback to random
        fiedler = np.random.randn(n)
    
    # Threshold at median to get balanced partition
    median_val = np.median(fiedler)
    labels = (fiedler > median_val).astype(np.int8)
    
    # Ensure both labels are present
    if labels.all() or not labels.any():
        labels = (np.random.rand(n) > 0.5).astype(np.int8)
    
    return labels


def _random_init(n):
    """Random balanced initialization."""
    labels = np.random.randint(0, 2, size=n)
    return labels


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 120 seconds total, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_labels = None
    best_cut = -1
    
    # Multiple initializations and local search
    inits = []
    
    # 1. Spectral initialization
    inits.append(_spectral_init(adj, n))
    
    # 2. Random initializations
    for _ in range(3):
        inits.append(_random_init(n))
    
    # 3. Alternating initialization
    inits.append(np.arange(n) % 2)
    
    for init_labels in inits:
        if time.time() > deadline - 5.0:
            break
        
        # Local search
        ls_labels = _local_search(adj, init_labels, max_iters=50)
        cut = _cut_size(adj, ls_labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = ls_labels.copy()
        
        # Simulated annealing from local search result
        if time.time() > deadline - 10.0:
            break
        
        sa_labels = _simulated_annealing(adj, ls_labels, start_temp=1.0, end_temp=0.01, cooling=0.99, max_iters=300)
        sa_cut = _cut_size(adj, sa_labels)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Another local search from SA result
        if time.time() > deadline - 5.0:
            break
        
        final_labels = _local_search(adj, sa_labels, max_iters=50)
        final_cut = _cut_size(adj, final_labels)
        
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
