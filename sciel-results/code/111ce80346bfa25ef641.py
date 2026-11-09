"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels is 0/1 vector
    # cut = sum_{v<u} adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # For a symmetric adjacency matrix:
    # cut = 0.5 * labels^T @ (D - A) @ labels  ... no that's not right
    # Actually: cut = sum_{v: label 0} sum_{u: label 1} adj[v,u]
    # Let's use: for each vertex v, count neighbors in opposite partition
    # cut = sum_v (adj[v] @ (1 - labels))  but this double counts
    # Actually adj[v] @ (1-labels) counts edges from v to vertices with label 1
    # If v has label 0, this is correct contribution
    # If v has label 1, this counts edges to label 0 vertices, also correct
    # So sum over all v of adj[v]@(1-labels) where v's label determines which side
    # Wait, let me think again.
    # cut = sum_{v in S} sum_{u in T} adj[v,u] where S={v: label=0}, T={v: label=1}
    # = sum_{v: label=0} sum_{u: label=1} adj[v,u]
    # = sum_v adj[v] @ (labels * (1-labels[v])) ... this is getting complicated
    # Simplest: cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # In numpy: diff = labels[:, None] != labels[None, :]  (boolean matrix)
    # cut = 0.5 * (adj * diff).sum()
    # But for speed with n=100, let's do:
    # For each vertex v with label 0: add sum of adj[v, labels==1]
    # For each vertex v with label 1: add sum of adj[v, labels==0]
    # This equals: sum_v adj[v] @ (1 - labels) if we're careful
    # Actually: sum_v adj[v] @ (1-labels) = sum_v sum_u adj[v,u]*(1-labels[u])
    # = sum_{v,u: labels[u]=0} adj[v,u]
    # This counts each edge (v,u) where u has label 0, regardless of v's label
    # If both have label 0: counted once (when v is the one with label 0... wait both are 0)
    # Hmm, let me just use the direct approach for n=100
    # cut = 0.5 * (adj * (labels[:, None] != labels[None, :])).sum()
    # But even simpler and faster:
    # cut = sum_{v in S} deg_to_T(v) where S is one side
    # Let me use: cut = (adj @ (1 - labels)).sum() / 2 ... no
    # 
    # Cleanest: 
    # Let x = 2*labels - 1 (so x is +1/-1)
    # cut = 0.25 * (n*|E| - x^T A x) ... not quite
    # 
    # For n=100, the boolean comparison is fine:
    diff = (labels[:, None] != labels[None, :])
    return int(0.5 * (adj * diff).sum())


def _spectral_init(adj, n, rng):
    """
    Use spectral embedding inspired by Goemans-Williamson.
    For Max-Cut, the SDP relaxation uses vectors v_i on the unit sphere.
    The spectral approach: compute the Laplacian L = D - A, find the 
    eigenvectors corresponding to the smallest non-trivial eigenvalues,
    and use their signs to form initial partitions.
    
    Actually, for Max-Cut, we want to maximize x^T (D-A) x / 4 subject to x_i = ±1.
    The relaxation replaces x with vectors. The spectral approach uses the
    Fiedler vector (2nd smallest eigenvector of L) or top eigenvectors of D-A.
    
    Let's use: compute eigenvalues/vectors of (D - A) = L.
    The largest eigenvalue of L corresponds to the "most cut-friendly" direction.
    Use the top few eigenvectors, project randomly, and take signs.
    """
    adj = np.asarray(adj, dtype=np.float64)
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj  # Laplacian
    
    # For Max-Cut, we want to maximize the cut. The spectral method uses
    # the eigenvector corresponding to the largest eigenvalue of (D-A) 
    # which is the same as the smallest non-zero eigenvalue of L... 
    # Actually, let's think: we want x^T L x to be large (since cut relates to this).
    # x^T L x = sum_{v<u} adj[v,u] (x_v - x_u)^2
    # For x in {+1,-1}, (x_v - x_u)^2 is 4 if different, 0 if same
    # So x^T L x = 4 * cut
    # We want to maximize x^T L x, so we want the eigenvector of L with LARGEST eigenvalue.
    
    # Compute eigenvalues and eigenvectors of L
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Use the top few eigenvectors (largest eigenvalues)
    # Take top 2-4 eigenvectors for a richer embedding
    n_vectors = min(4, n)
    top_eigvecs = eigvecs[:, -n_vectors:]  # largest eigenvalues
    
    # Random hyperplane projection: pick a random vector in the embedding space
    # and project, then take signs
    # Or: use the top eigenvector directly
    # Let's try: use a random linear combination of the top eigenvectors
    weights = rng.standard_normal(n_vectors)
    v = top_eigvecs @ weights
    
    # Take signs
    labels = (v >= 0).astype(np.int64)
    
    return labels


def _greedy_flip_improve(adj, labels):
    """Greedy local improvement: flip vertex if it improves the cut."""
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.copy()
    
    # Precompute for efficiency
    # For each vertex v, the gain from flipping:
    # If v is in set 0: gain = (# neighbors in set 1) - (# neighbors in set 0)
    # If v is in set 1: gain = (# neighbors in set 0) - (# neighbors in set 1)
    # = (degree[v] - 2 * (# neighbors in same set))
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            same_count = adj[v][labels == labels[v]].sum()
            diff_count = adj[v][labels != labels[v]].sum()
            if diff_count > same_count:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _two_swap_improve(adj, labels, max_iter=10):
    """2-swap local search: flip pairs of vertices if it improves the cut."""
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    for _ in range(max_iter):
        improved = False
        # Try all pairs (expensive but n=100 so 4950 pairs)
        # For efficiency, only try pairs where at least one might help
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue  # Swapping same-side vertices doesn't change cut
                # Try swapping i and j
                labels[i], labels[j] = labels[j], labels[i]
                new_cut = _cut_size(adj, labels)
                if new_cut > current_cut:
                    current_cut = new_cut
                    improved = True
                else:
                    labels[i], labels[j] = labels[j], labels[i]
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return labels


def partition(adjacency):
    """
    Maximum Cut via spectral initialization + local search with 2-swaps.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 140s out of 150s
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy: multiple spectral-based initializations + local search
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial partition
        if restart == 0:
            # First: alternating pattern
            labels = np.arange(n) % 2
        elif restart == 1:
            # Second: random
            labels = np_rng.integers(0, 2, n)
        else:
            # Spectral initialization with random projection
            labels = _spectral_init(adj, n, np_rng)
        
        # Greedy 1-flip improvement
        labels = _greedy_flip_improve(adj, labels)
        cut_val = _cut_size(adj, labels)
        
        # 2-swap improvement
        if time.time() < deadline - 5.0:
            labels = _two_swap_improve(adj, labels, max_iter=5)
            cut_val = _cut_size(adj, labels)
        
        # Final greedy polish
        labels = _greedy_flip_improve(adj, labels)
        cut_val = _cut_size(adj, labels)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels.copy()
    
    # If we still have time, do more restarts with perturbation
    while time.time() < deadline - 2.0:
        if best_labels is None:
            break
        
        # Perturb best solution
        labels = best_labels.copy()
        # Flip 2-5 random vertices
        n_flips = rng.randint(2, 5)
        flip_idx = rng.sample(range(n), n_flips)
        for idx in flip_idx:
            labels[idx] = 1 - labels[idx]
        
        labels = _greedy_flip_improve(adj, labels)
        if time.time() < deadline - 2.0:
            labels = _two_swap_improve(adj, labels, max_iter=3)
        labels = _greedy_flip_improve(adj, labels)
        cut_val = _cut_size(adj, labels)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
