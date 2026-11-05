"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Fixed safety margin so we return well before the 150 s wall-clock kill.
DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size for a given labeling."""
    diff = (labels[None, :] != labels[:, None])
    return int(np.sum(adj * diff))


def _spectral_init(adj):
    """
    Compute the Fiedler vector of the graph Laplacian to get an initial partition.
    """
    n = len(adj)
    # Build Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = sp.diags(deg) - sp.csr_matrix(adj)
    
    # Find the Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
    # Use shift-invert to find small eigenvalues
    try:
        # We want eigenvalues near 0, so use shift-invert with sigma=0
        vals, vecs = spla.eigsh(L, k=2, sigma=0, which='LM', return_eigenvectors=True)
        # The second smallest eigenvalue corresponds to the Fiedler vector
        # eigsh returns in ascending order of distance from sigma, so we need to sort
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    except Exception:
        # Fallback to random if eigsh fails
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(n)
    
    # Partition by sign of Fiedler vector
    labels = (fiedler >= 0).astype(int)
    return labels


def _vectorized_gains(labels, adj):
    """
    Compute flip gains for all vertices in a vectorized manner.
    gain[v] = 2 * (deg(v) - same_side_neighbors[v]) - deg(v)
             = 2 * other_side_neighbors[v] - deg(v)
    """
    n = len(labels)
    # same_side_neighbors[v] = sum of adj[v, u] where labels[u] == labels[v]
    # We can compute this using matrix multiplication
    # Let's define: for each v, same_count[v] = sum_u adj[v,u] * (labels[u] == labels[v])
    
    # Create a matrix where M[v,u] = 1 if labels[u] == labels[v], else 0
    # This is equivalent to: M = (labels[None,:] == labels[:,None])
    same_mask = (labels[None, :] == labels[:, None])
    
    # same_counts[v] = sum_u adj[v,u] * same_mask[v,u]
    same_counts = (adj * same_mask).sum(axis=1)
    deg = adj.sum(axis=1)
    
    gains = 2 * (deg - same_counts) - deg
    return gains


def _greedy_polish(labels, adj, deadline):
    """
    Greedy local search with vectorized gain computation.
    Repeatedly flip the vertex with the best positive gain.
    """
    n = len(labels)
    labels = labels.copy()
    
    while True:
        if time.time() > deadline:
            break
        
        gains = _vectorized_gains(labels, adj)
        best_v = np.argmax(gains)
        best_gain = gains[best_v]
        
        if best_gain <= 0:
            break
        
        labels[best_v] = 1 - labels[best_v]
    
    return labels


def _pairwise_swap_polish(labels, adj, deadline):
    """
    Local search with pairwise swap moves to escape single-flip local optima.
    A swap exchanges labels of two vertices.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        
        improved = False
        best_swap_gain = 0
        best_pair = None
        
        # Compute gains for all pairs would be O(n^2 * n) which is too slow for n=100
        # Instead, we can be smarter: only consider pairs where one is 0 and other is 1
        # For a swap of i and j:
        #   i: label a -> 1-a
        #   j: label b -> 1-b
        #   where a != b (otherwise no change)
        
        # Let's use a heuristic: try random pairs or all pairs if n is small
        # For n=100, all pairs is 4950, and computing gain for each is O(deg(i)+deg(j))
        # Total per iteration: ~4950 * 10 = ~50k ops, which is fast enough
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                
                # Compute gain of swapping i and j
                # Before swap:
                #   i contributes to cut: edges from i to nodes with label != labels[i]
                #   j contributes to cut: edges from j to nodes with label != labels[j]
                
                # After swap:
                #   i has label 1-labels[i] = labels[j]
                #   j has label 1-labels[j] = labels[i]
                
                # The change in cut size:
                # For edges involving i or j (but not both i and j directly counted twice)
                
                # Let's compute directly:
                # cut_before = cut involving i + cut involving j - (edge i-j if exists and crosses)
                # Actually, let's just compute the gain by considering all neighbors
                
                # Gain from flipping i alone (if j didn't exist) + gain from flipping j alone
                # minus the interaction term (edge between i and j)
                
                # Simpler: compute the actual cut contribution change
                # For each neighbor k of i:
                #   Before: edge (i,k) crosses if labels[k] != labels[i]
                #   After:  edge (i,k) crosses if labels[k] != labels[j] (since i now has label of j)
                # For each neighbor k of j:
                #   Before: edge (j,k) crosses if labels[k] != labels[j]
                #   After:  edge (j,k) crosses if labels[k] != labels[i] (since j now has label of i)
                
                # Net change = sum over k in N(i) of [1(labels[k]!=labels[j]) - 1(labels[k]!=labels[i])]
                #           + sum over k in N(j) of [1(labels[k]!=labels[i]) - 1(labels[k]!=labels[j])]
                
                # Note: if k is both in N(i) and N(j), and k != i, j, the terms might cancel partially
                # But since we're summing over neighbors separately, and the edge (i,j) is handled:
                #   For k=j in N(i): before labels[j]!=labels[i] (true since different), after labels[j]!=labels[j] (false) -> change -1
                #   For k=i in N(j): before labels[i]!=labels[j] (true), after labels[i]!=labels[i] (false) -> change -1
                # So edge (i,j) contributes -2 to the gain if it exists.
                
                ni = adj[i, :]
                nj = adj[j, :]
                
                # Vectorized computation
                # Term 1: sum over k where ni[k]=1 of [1(labels[k]!=labels[j]) - 1(labels[k]!=labels[i])]
                # = sum_k ni[k] * (1(labels[k]!=labels[j]) - 1(labels[k]!=labels[i]))
                
                diff_ij = labels[i] != labels[j]  # True
                # 1(labels[k]!=labels[j]) - 1(labels[k]!=labels[i])
                # If labels[i]=0, labels[j]=1:
                #   1(labels[k]!=1) - 1(labels[k]!=0) = labels[k] - (1-labels[k]) = 2*labels[k] - 1
                # If labels[i]=1, labels[j]=0:
                #   1(labels[k]!=0) - 1(labels[k]!=1) = (1-labels[k]) - labels[k] = 1 - 2*labels[k]
                
                if labels[i] == 0:
                    term1 = np.sum(ni * (2 * labels - 1))
                else:
                    term1 = np.sum(ni * (1 - 2 * labels))
                
                if labels[j] == 0:
                    term2 = np.sum(nj * (2 * labels - 1))
                else:
                    term2 = np.sum(nj * (1 - 2 * labels))
                
                swap_gain = term1 + term2
                
                if swap_gain > best_swap_gain:
                    best_swap_gain = swap_gain
                    best_pair = (i, j)
        
        if best_pair is not None and best_swap_gain > 0:
            i, j = best_pair
            labels[i], labels[j] = labels[j], labels[i]
            improved = True
    
    return labels


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the cut size.
    Uses spectral initialization followed by multi-start local search with
    greedy flip and pairwise swap moves.
    """
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=bool)
    n = len(adj)

    best_labels = None
    best_cut = -1

    # Multiple initializations
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Spectral initialization
    try:
        spectral_labels = _spectral_init(adj)
        initializations.append(spectral_labels)
    except Exception:
        pass
    
    # Alternating
    initializations.append(np.arange(n) % 2)
    
    # Random initializations
    for _ in range(10):
        initializations.append(rng.integers(0, 2, size=n))

    for init_labels in initializations:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # Phase 1: Greedy flip polish
        labels = _greedy_polish(labels, adj, deadline)
        
        # Phase 2: Pairwise swap polish
        labels = _pairwise_swap_polish(labels, adj, deadline)
        
        # Phase 3: Another greedy flip polish after swaps
        labels = _greedy_polish(labels, adj, deadline)
        
        cut = _cut_size(labels, adj)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # If we have time, do a few more random restarts
    while time.time() < deadline - 10:
        init_labels = rng.integers(0, 2, size=n)
        labels = init_labels.copy()
        labels = _greedy_polish(labels, adj, deadline)
        labels = _pairwise_swap_polish(labels, adj, deadline)
        labels = _greedy_polish(labels, adj, deadline)
        cut = _cut_size(labels, adj)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
