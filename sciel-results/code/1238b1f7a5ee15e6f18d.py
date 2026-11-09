# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels is a 0/1 array of length n
    # cut = sum_{v} sum_{u: labels[u] != labels[v]} adj[v,u]
    # Using matrix ops: for each v, count neighbors on opposite side
    # opposite = 1 - labels
    # cut = sum_v (adj[v] @ opposite)
    # But this counts each edge twice (once from each endpoint), so divide by 2
    opposite = 1.0 - labels.astype(float)
    cut = (adj.astype(float) @ opposite).sum() / 2.0
    return int(round(cut))


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (second eigenvector) of the Laplacian."""
    A = adj.astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Use scipy if available for better eigensolver, otherwise numpy
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L_sp = csr_matrix(L)
        # Find the smallest eigenvalue (0) and second smallest
        # eigsh finds eigenvalues near sigma
        vals, vecs = eigsh(L_sp, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    except Exception:
        # Fallback to numpy
        vals, vecs = np.linalg.eigh(L)
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    
    # Threshold at 0 (or median) to get 0/1 labels
    labels = (fiedler > 0).astype(int)
    # Ensure both partitions are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    return labels


def _greedy_flip(adj, labels, max_iters=10):
    """Greedy local improvement: flip a vertex if it increases the cut."""
    n = len(labels)
    A = adj.astype(float)
    labels_f = labels.astype(float)
    
    for _ in range(max_iters):
        improved = False
        # For each vertex, compute the gain from flipping
        # gain(v) = 2 * (neighbors on opposite side) - degree(v)
        # Actually: if we flip v, the change in cut is:
        # new_edges_crossing - old_edges_crossing
        # = (deg(v) - same_side) - same_side  ... wait
        # Let same = number of neighbors of v on same side
        # Let other = number of neighbors of v on opposite side
        # deg = same + other
        # If we flip v:
        #   Edges that were crossing (v to other) become non-crossing: -other
        #   Edges that were not crossing (v to same) become crossing: +same
        #   Net change: same - other
        # So flip if same > other, i.e., same - other > 0
        # same = A[v] @ labels_f (sum of neighbors on same side, since labels_f is 0/1)
        # other = deg[v] - same
        # gain = same - other = 2*same - deg[v]
        
        deg = A.sum(axis=1)
        same = A @ labels_f  # for each v, sum of adj[v,u] where labels[u] == 1
        # But we need same-side count: if labels[v] == 1, same = A[v] @ labels_f
        # if labels[v] == 0, same = A[v] @ (1 - labels_f)
        # Let's compute properly:
        # same_side[v] = sum_u adj[v,u] * (labels[u] == labels[v])
        # = sum_u adj[v,u] * (labels[u] * labels[v] + (1-labels[u]) * (1-labels[v]))
        # This is getting complex. Let's just compute gain directly.
        
        # gain[v] = 2 * (neighbors on opposite side) - deg[v]
        # neighbors on opposite side for v:
        # if labels[v] == 0: opposite = 1, so count neighbors with label 1
        # if labels[v] == 1: opposite = 0, so count neighbors with label 0
        opp = 1.0 - labels_f
        # neighbors on opposite side = A @ opp ... but this gives for each v, sum of adj[v,u] * opp[u]
        # which is exactly the number of neighbors on the opposite side (since opp[u] = 1 iff labels[u] != labels[v])
        opp_count = A @ opp
        gain = 2.0 * opp_count - deg
        
        # Find the vertex with the largest positive gain
        max_gain = gain.max()
        if max_gain > 1e-9:
            v = int(np.argmax(gain))
            labels[v] = 1 - labels[v]
            labels_f[v] = 1.0 - labels_f[v]
            improved = True
        if not improved:
            break
    
    return labels


def _two_swap(adj, labels, max_iters=20):
    """Try 2-swap moves: flip two vertices simultaneously if it improves the cut."""
    n = len(labels)
    A = adj.astype(float)
    labels_f = labels.astype(float)
    
    for _ in range(max_iters):
        improved = False
        # For a 2-swap of vertices i and j:
        # The change in cut depends on the interactions.
        # This is more complex. Let's try a simpler approach:
        # Randomly pick pairs and check if flipping both improves the cut.
        # For efficiency, we'll do a limited random search.
        
        # Compute current cut
        opp = 1.0 - labels_f
        current_cut = (A @ opp).sum() / 2.0
        
        # Try a few random pairs
        for _ in range(50):
            i, j = random.sample(range(n), 2)
            # Temporarily flip i and j
            labels_f[i] = 1.0 - labels_f[i]
            labels_f[j] = 1.0 - labels_f[j]
            opp_new = 1.0 - labels_f
            new_cut = (A @ opp_new).sum() / 2.0
            if new_cut > current_cut + 1e-9:
                labels[i] = int(labels_f[i])
                labels[j] = int(labels_f[j])
                current_cut = new_cut
                improved = True
                break
            else:
                # Revert
                labels_f[i] = 1.0 - labels_f[i]
                labels_f[j] = 1.0 - labels_f[j]
        
        if not improved:
            break
    
    return labels


def partition(adjacency):
    """Partition the graph into two sets to maximize the cut size."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 140s out of 150s limit
    
    adj = np.array(adjacency, dtype=bool)
    n = len(adj)
    A = adj.astype(float)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if r == 0:
            # First restart: spectral initialization
            labels = _spectral_init(adj, n)
        elif r == 1:
            # Second restart: random initialization
            labels = np.random.randint(0, 2, size=n)
        else:
            # Subsequent restarts: random with some bias
            labels = np.random.randint(0, 2, size=n)
        
        # Greedy flip improvement
        labels = _greedy_flip(adj, labels, max_iters=20)
        
        # 2-swap improvement
        if time.time() < deadline - 10.0:
            labels = _two_swap(adj, labels, max_iters=30)
        
        # Final greedy polish
        labels = _greedy_flip(adj, labels, max_iters=10)
        
        # Evaluate
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If no labels were found (shouldn't happen), use a default
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
