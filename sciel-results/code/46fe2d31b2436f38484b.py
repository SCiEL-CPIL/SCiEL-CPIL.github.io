"""Maximum Cut: Spectral initialization + parallel greedy local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2


def _spectral_init(adj, n, rng):
    """Spectral initialization using the Fiedler vector of the graph Laplacian."""
    # Build Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    # Compute eigenvector corresponding to the second smallest eigenvalue (Fiedler vector)
    # For efficiency, use scipy if available, otherwise use power iteration on shifted matrix
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import eigsh
        L_sp = csc_matrix(L)
        # Find the two smallest eigenvalues/eigenvectors
        vals, vecs = eigsh(L_sp, k=2, which='SM')
        # The Fiedler vector is the eigenvector for the second smallest eigenvalue
        fiedler = vecs[:, 1]
    except Exception:
        # Fallback: use random projection or simple alternating
        fiedler = rng.standard_normal(n)
    
    # Partition based on the sign of the Fiedler vector (or median split)
    median_val = np.median(fiedler)
    labels = (fiedler > median_val).astype(np.int8)
    return labels


def _greedy_1_flip(adj, labels, deadline, max_rounds=100):
    """Greedy 1-flip local search: flip a vertex if it increases cut size."""
    n = len(labels)
    # Precompute: for each vertex, the gain of flipping it
    # Gain of flipping v = 2 * (neighbors in other group) - degree(v)
    # More precisely: if v is in group 0, gain = (adj[v] & (labels==1)).sum() - (adj[v] & (labels==0)).sum()
    #                  if v is in group 1, gain = (adj[v] & (labels==0)).sum() - (adj[v] & (labels==1)).sum()
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Compute gains for all vertices at once
        # For vertex v in group 0: gain = (adj[v] & (labels==1)).sum() - (adj[v] & (labels==0)).sum()
        # For vertex v in group 1: gain = (adj[v] & (labels==0)).sum() - (adj[v] & (labels==1)).sum()
        
        labels_in_0 = (labels == 0).astype(np.int8)
        labels_in_1 = (labels == 1).astype(np.int8)
        
        # For each vertex v, count neighbors in group 0 and group 1
        neigh_in_0 = adj @ labels_in_0  # number of neighbors in group 0
        neigh_in_1 = adj @ labels_in_1  # number of neighbors in group 1
        degrees = adj.sum(axis=1)
        
        # Gain if we flip v:
        # If v is in group 0: currently contributes (neigh_in_1[v]) crossing edges.
        # After flip, contributes (neigh_in_0[v]) crossing edges (but v is now in group 1, so neighbors in group 0 are now crossing)
        # Wait, let me think more carefully.
        # If v is in group 0:
        #   Current crossing edges from v: edges to neighbors in group 1 = neigh_in_1[v]
        #   After flip (v in group 1): crossing edges from v = edges to neighbors in group 0 = neigh_in_0[v]
        #   But note: when we flip v, the edge (v,u) for u in group 0 was NOT crossing, now it IS crossing.
        #   The edge (v,u) for u in group 1 WAS crossing, now it is NOT crossing.
        #   So change = neigh_in_0[v] - neigh_in_1[v]
        # If v is in group 1:
        #   Current crossing edges from v: edges to neighbors in group 0 = neigh_in_0[v]
        #   After flip (v in group 0): crossing edges from v = edges to neighbors in group 1 = neigh_in_1[v]
        #   So change = neigh_in_1[v] - neigh_in_0[v]
        
        # Gain for flipping v:
        gains = np.where(labels == 0, neigh_in_0 - neigh_in_1, neigh_in_1 - neigh_in_0)
        
        # Find vertices with positive gain
        positive_gains = gains > 0
        if not positive_gains.any():
            break
        
        # Flip all vertices with positive gain simultaneously (parallel flip)
        labels[positive_gains] = 1 - labels[positive_gains]
        improved = True
    
    return labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # First try spectral initialization
    labels = _spectral_init(adj, n, rng)
    labels = _greedy_1_flip(adj, labels.copy(), deadline)
    cut = _cut_size(adj, labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Then try random restarts with greedy local search
    while time.time() < deadline:
        # Random initialization
        labels = rng.integers(0, 2, size=n).astype(np.int8)
        
        # Greedy local improvement
        labels = _greedy_1_flip(adj, labels.copy(), deadline)
        
        if time.time() > deadline:
            break
        
        # Evaluate this solution
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we never got a chance to run, use alternating start
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels = labels
    
    return best_labels
# EVOLVE-BLOCK-END
