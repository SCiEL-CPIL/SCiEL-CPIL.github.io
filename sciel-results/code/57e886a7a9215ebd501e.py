"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # For each vertex, count edges to vertices with different label
    # Using matrix multiplication for speed
    # cut = sum over v of sum over u where labels[v] != labels[u] of adj[v,u]
    # Equivalent to: 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # But we can compute more efficiently:
    # Let s_v = sum_u adj[v,u] * (labels[u] == labels[v]) ... no
    # Actually: cut = sum_{v < u} adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * sum_{v,u} adj[v,u] * (labels[v] != labels[u])
    
    # Efficient: for each v, count neighbors with opposite label
    # Using the fact that adj is symmetric (undirected graph)
    # cut = 0.5 * sum_v (sum_u adj[v,u]) - sum_v sum_u adj[v,u] * (labels[v] == labels[u])
    # Hmm, let me think differently.
    
    # Let's use: for vertex v with label l, the number of edges from v to opposite side
    # is: sum_u adj[v,u] - sum_u adj[v,u] * (labels[u] == labels[v])
    # Total cut = 0.5 * sum_v [sum_u adj[v,u] - sum_u adj[v,u] * (labels[u] == labels[v])]
    
    # Even simpler: 
    # Let A be adjacency matrix, x be label vector in {0,1}
    # cut = 0.5 * sum_{v,u} A[v,u] * (x[v] != x[u])
    # = 0.5 * sum_{v,u} A[v,u] * (1 - (2*x[v]-1)*(2*x[u]-1))  ... no that's for ±1
    # For 0/1: (x[v] != x[u]) = x[v] + x[u] - 2*x[v]*x[u]
    # So cut = 0.5 * sum_{v,u} A[v,u] * (x[v] + x[u] - 2*x[v]*x[u])
    # = 0.5 * [sum_v x[v] * (A@1)[v] + sum_u x[u] * (A@1)[u] - 2 * x^T A x]
    # = 0.5 * [2 * x^T (A@1) - 2 * x^T A x]
    # = x^T (A@1) - x^T A x
    
    # But wait, this counts each edge twice in the first term? Let me verify.
    # Actually for undirected graph, A is symmetric, so sum_{v,u} A[v,u]*x[v] = sum_v x[v] * sum_u A[v,u] = x^T (A @ ones)
    # And sum_{v,u} A[v,u]*x[v]*x[u] = x^T A x
    # So cut = x^T (A @ ones) - x^T A x
    
    # Let me just compute it directly but efficiently
    # For each v, same_side_count = sum_u adj[v,u] * (labels[u] == labels[v])
    # cut = 0.5 * sum_v (degree[v] - same_side_count[v])
    
    # Vectorized approach:
    # same_side = for each v, sum of adj[v,u] where labels[u] == labels[v]
    # We can compute: for label 0 vertices, sum adj[v, :][labels==0]
    # for label 1 vertices, sum adj[v, :][labels==1]
    
    # Even better: use matrix ops
    # Let's create a matrix M where M[v,u] = 1 if labels[v] != labels[u], else 0
    # cut = 0.5 * (A * M).sum()
    
    # But creating M explicitly is O(n^2). For n=100 that's fine.
    diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    return 0.5 * (adj * diff).sum()


def _local_search(adj, labels, deadline, max_iter=50000):
    """Greedy local search: flip a vertex if it improves the cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # For each vertex v, the contribution to cut from v's edges:
    # edges from v to opposite side
    # When we flip v, the change in cut is:
    # new_cut - old_cut = (edges from v to same side) - (edges from v to opposite side)
    # = same_side_edges - opp_side_edges
    # where same_side_edges = sum_u adj[v,u] * (labels[u] == labels[v])
    #       opp_side_edges = sum_u adj[v,u] * (labels[u] != labels[v])
    # And same_side + opp_side = degree[v]
    # So change = same_side - opp_side = 2*same_side - degree[v]
    # We flip if change > 0, i.e., same_side > opp_side, i.e., 2*same_side > degree[v]
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        if time.time() > deadline:
            break
        for v in range(n):
            if time.time() > deadline:
                break
            # Compute same_side edges for vertex v
            same_mask = (labels == labels[v])
            same_side = adj[v, same_mask].sum()
            opp_side = degrees[v] - same_side
            if same_side > opp_side:
                labels[v] = 1 - labels[v]
                current_cut += (same_side - opp_side)
                improved = True
                iter_count += 1
    return labels, current_cut


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization + local search + restarts.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline: leave 10s safety margin before 120s limit
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # Strategy: multiple restarts with different initializations
    # 1. Spectral initialization (Fiedler vector)
    # 2. Random initializations
    # 3. Alternating initialization
    
    # Try spectral initialization first
    try:
        # Compute Laplacian L = D - A
        D = np.diag(adj.sum(axis=1))
        L = D - adj
        
        # Find eigenvector corresponding to smallest non-zero eigenvalue
        # For speed, use scipy if available, otherwise numpy
        try:
            from scipy.linalg import eigh
            # Get eigenvalues and eigenvectors of L
            eigvals, eigvecs = eigh(L, subset_by_index=[1, 1])
            fiedler = eigvecs[:, 0]
        except ImportError:
            # Fallback to numpy
            eigvals, eigvecs = np.linalg.eigh(L)
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
        
        # Threshold at median
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(np.int32)
        
        labels, cut = _local_search(adj, labels, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    except Exception:
        pass
    
    # Try alternating initialization
    labels = np.arange(n) % 2
    labels, cut = _local_search(adj, labels, deadline)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Try random restarts with remaining time
    rng = np.random.RandomState(42)
    while time.time() < deadline - 5.0:
        labels = rng.randint(0, 2, size=n).astype(np.int32)
        labels, cut = _local_search(adj, labels, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final answer
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
