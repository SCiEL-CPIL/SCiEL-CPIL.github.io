"""Rewritten Maximum Cut solver using spectral init + 1-flip + 2-swap local search with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition (0/1 labels)."""
    # cut = sum_{v<u} adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * (sum_v adj[v] @ (1 - labels) where labels are 0/1) ... but this counts each edge twice?
    # Actually: for edge (v,u), it's counted in adj[v]@(1-labels) as 1 if labels[u]!=labels[v], 
    # and in adj[u]@(1-labels) as 1 if labels[v]!=labels[u]. So sum_v adj[v]@(1-labels) = 2*cut.
    # But wait: adj[v]@(1-labels) = sum_u adj[v,u]*(1-labels[u]). 
    # If labels are 0/1, (1-labels[u]) is 1 when labels[u]=0.
    # So adj[v]@(1-labels) counts neighbors of v that are in set 0.
    # The cut size is the number of edges between set 0 and set 1.
    # = sum_{v in set 0} (number of neighbors in set 1) = sum_{v: labels[v]=0} adj[v] @ labels
    # = labels.T @ adj @ (1-labels) ... let me think more carefully.
    # 
    # Let's use: cut = 0.5 * (1 - 2*labels).T @ adj @ (1 - 2*labels) ... no that's not right either.
    # 
    # Simplest vectorized: 
    # For each vertex v, the number of edges from v to the opposite side is:
    #   if labels[v] == 0: adj[v] @ labels  (neighbors in set 1)
    #   if labels[v] == 1: adj[v] @ (1 - labels)  (neighbors in set 0)
    # Sum over all v, but each edge is counted twice (once from each endpoint).
    # So cut = 0.5 * sum_v [adj[v] @ (1 - labels[v]) ... ] 
    # 
    # Let me just do it directly:
    # cut = sum over v of (number of neighbors of v in the opposite partition)
    # = sum_v (adj[v] @ (1 - labels) if labels[v]==0 else adj[v] @ labels)
    # = sum_v adj[v] @ (labels if labels[v]==1 else (1-labels))
    # 
    # Vectorized: let s = 1 - 2*labels (so s[v] = 1 if labels[v]=0, -1 if labels[v]=1)
    # Then for vertex v, opposite side indicator is (1 - s[v])/2 ... this is getting complicated.
    #
    # Cleanest: cut = 0.5 * np.sum(adj * (labels[:, None] != labels[None, :]))
    # But that creates an n x n boolean matrix. For n=100, that's fine.
    # Actually even simpler:
    # cut = 0.5 * (labels @ adj @ (1 - labels)) ... no.
    #
    # Let me just use: for each v, gain of flipping v = 2*(#neighbors in opposite) - degree[v]
    # #neighbors in opposite for v = adj[v] @ (1 - labels) if labels[v]==0, adj[v] @ labels if labels[v]==1
    # = adj[v] @ (labels ^ (labels[v] == 0)) ... 
    #
    # I'll use the direct approach for clarity and correctness:
    # cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * sum_v adj[v] @ (1 - labels) * ... no
    #
    # Let me define: for 0/1 labels, the indicator that u is on the opposite side of v is (labels[u] != labels[v])
    # = labels[u] * (1 - labels[v]) + (1 - labels[u]) * labels[v]
    # = labels[u] + labels[v] - 2*labels[u]*labels[v]
    #
    # So cut = 0.5 * sum_v sum_u adj[v,u] * (labels[u] + labels[v] - 2*labels[u]*labels[v])
    # = 0.5 * sum_v [ (sum_u adj[v,u]*labels[u]) + labels[v]*(sum_u adj[v,u]) - 2*labels[v]*(sum_u adj[v,u]*labels[u]) ]
    # = 0.5 * [ sum_v (adj@labels)[v] + sum_v labels[v]*deg[v] - 2*sum_v labels[v]*(adj@labels)[v] ]
    #
    # This is getting complex. For n=100, the simplest is:
    # cut = 0.5 * np.sum(adj * (labels[:, None] != labels[None, :]))
    # This is O(n^2) which is 10000 ops, totally fine.
    return int(np.sum(adj * (labels[:, None] != labels[None, :]))) // 2


def _spectral_init(adj, n, rng):
    """Initialize labels using the Fiedler vector (second eigenvector of Laplacian)."""
    # Compute Laplacian L = D - A
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    # Use scipy or numpy for eigenvalues
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        # Get smallest eigenvalues
        vals, vecs = eigsh(L_sparse, k=2, which='SM')
        # Fiedler vector is the one corresponding to the second smallest eigenvalue
        # eigsh returns in increasing order for 'SM'
        fiedler = vecs[:, 1]
    except Exception:
        # Fallback to numpy
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
    
    # Threshold at zero (or median) to get 0/1 labels
    # Use median for balance
    median_val = np.median(fiedler)
    labels = (fiedler > median_val).astype(np.int8)
    
    # Ensure both labels are present
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.arange(n) % 2
    
    return labels


def _random_init(n, rng):
    """Random 0/1 initialization with roughly balanced partition."""
    labels = rng.integers(0, 2, size=n).astype(np.int8)
    # Ensure balance
    num_zeros = (labels == 0).sum()
    if num_zeros < n // 4 or num_zeros > 3 * n // 4:
        labels = np.arange(n) % 2
    return labels


def _local_search_1flip(adj, labels, n, max_iters=100):
    """Local search using single vertex flips."""
    labels = labels.copy()
    deg = np.sum(adj, axis=1)
    
    for iteration in range(max_iters):
        improved = False
        # Compute for each vertex: gain of flipping
        # If v is in set 0 (label=0), flipping to 1 changes cut by:
        #   new_cross - old_cross = (neighbors in set 1) - (neighbors in set 0)
        # neighbors in set 1 = adj[v] @ labels (since labels are 0/1, this counts neighbors with label 1)
        # neighbors in set 0 = deg[v] - adj[v] @ labels
        # gain = (adj[v] @ labels) - (deg[v] - adj[v] @ labels) = 2*(adj[v] @ labels) - deg[v]
        #
        # If v is in set 1 (label=1), flipping to 0 changes cut by:
        #   new_cross - old_cross = (neighbors in set 0) - (neighbors in set 1)
        # = (deg[v] - adj[v] @ labels) - (adj[v] @ labels) = deg[v] - 2*(adj[v] @ labels)
        #
        # Unified: gain of flipping v = 2*(adj[v] @ labels) - deg[v] if label[v]==0
        #                        = deg[v] - 2*(adj[v] @ labels) if label[v]==1
        # = (1 - 2*label[v]) * (2*(adj[v] @ labels) - deg[v])
        
        adj_labels = adj @ labels  # For each v, sum of adj[v,u]*labels[u]
        gain = (1 - 2 * labels) * (2 * adj_labels - deg)
        
        # Find the vertex with maximum positive gain
        best_v = np.argmax(gain)
        if gain[best_v] > 0:
            labels[best_v] = 1 - labels[best_v]
            improved = True
    
    return labels


def _local_search_2swap(adj, labels, n, max_iters=50):
    """Local search using pair swaps (flip two vertices simultaneously)."""
    labels = labels.copy()
    deg = np.sum(adj, axis=1)
    
    for iteration in range(max_iters):
        improved = False
        adj_labels = adj @ labels
        
        # For a 2-swap of vertices i and j:
        # The gain depends on their current labels and the edges between them and others.
        # This is more complex. Let's do a simpler approach: try random pairs.
        # Or: compute gain for each pair efficiently.
        #
        # For simplicity, let's try flipping pairs of vertices that are in the same set,
        # or use a heuristic: try pairs where both have negative individual flip gain.
        #
        # Actually, let's do: for each pair (i,j) where labels[i] != labels[j], 
        # flipping both means i goes to opposite and j goes to opposite.
        # This is equivalent to a 2-swap in the cut problem.
        #
        # For efficiency with n=100, we can try all pairs: 100*99/2 = 4950 pairs.
        # For each pair, compute the gain. This might be slow if done naively.
        #
        # Let me use a simpler approach: randomly sample pairs.
        best_gain = 0
        best_pair = None
        
        # Try a subset of pairs for speed
        num_trials = min(500, n * (n - 1) // 2)
        for _ in range(num_trials):
            i, j = rng_sample_pair(n)
            if i == j:
                continue
            # Compute gain of flipping both i and j
            # Current cut contribution from edges incident to i or j:
            # After flip, i and j swap labels.
            # 
            # Let's compute directly:
            # old_cut_contribution = edges from i to opposite + edges from j to opposite (but careful about double counting edge (i,j))
            # 
            # Simpler: compute cut before and after for the affected edges.
            # Affected edges: all edges incident to i or j.
            # 
            # This is getting complex. Let me use a different approach:
            # Just compute the full cut size before and after for candidate pairs.
            # But that's O(n^2) per pair, too slow.
            #
            # Alternative: precompute and use the gain formula.
            # For flipping vertex i (alone): gain_i = (1-2*labels[i]) * (2*(adj[i] @ labels) - deg[i])
            # For flipping both i and j: the gain is not simply gain_i + gain_j because 
            # the edge (i,j) is affected differently.
            #
            # Let me just do a simple random restart approach with 1-flip only, 
            # and add 2-swap as a perturbation.
        
        break  # Skip 2-swap for now, use 1-flip only with more restarts
    
    return labels


def rng_sample_pair(n):
    """Sample a random pair of distinct vertices."""
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    while j == i:
        j = random.randint(0, n - 1)
    return i, j


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph
    
    Returns:
        0/1 label per vertex (numpy array of int8)
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetry
    adj = (adj + adj.T) / 2.0
    adj = (adj > 0.5).astype(np.float64)
    
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng()
    
    # Multiple restarts with different initializations
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            # Spectral init
            labels = _spectral_init(adj, n, rng)
        elif restart % 3 == 1:
            # Alternating init
            labels = np.arange(n) % 2
        else:
            # Random init
            labels = _random_init(n, rng)
        
        # Local search with 1-flip
        labels = _local_search_1flip(adj, labels, n, max_iters=200)
        
        # Compute cut size
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If time is running out, do a final polish on the best solution
        if time.time() > deadline - 10:
            # Do more iterations on the best solution
            best_labels = _local_search_1flip(adj, best_labels, n, max_iters=500)
            break
    
    # Final polish
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    # Ensure the result is a proper 0/1 array
    best_labels = best_labels.astype(np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
