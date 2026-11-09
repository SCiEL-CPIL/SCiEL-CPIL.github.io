# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _flip_gain(adj, labels, v):
    """Compute the gain from flipping vertex v.
    Gain = (neighbors in same set) - (neighbors in opposite set)
    """
    same_mask = (labels == labels[v])
    opp_mask = (labels != labels[v])
    same_count = adj[v][same_mask].sum()
    opp_count = adj[v][opp_mask].sum()
    return same_count - opp_count


def _two_swap_gain(adj, labels, i, j):
    """Compute the gain from simultaneously flipping vertices i and j.
    Only valid if i != j and labels[i] != labels[j] (otherwise no change).
    """
    if i == j or labels[i] == labels[j]:
        return 0
    # Flipping both i and j:
    # Edge (i,j): was crossing (labels differ), after flip still crossing -> no change
    # Edges from i to others (excluding j):
    #   For u != i,j: if labels[i]==labels[u], after flip labels[i]!=labels[u] -> +1
    #                 if labels[i]!=labels[u], after flip labels[i]==labels[u] -> -1
    #   Same for j.
    # Net gain = sum over u not in {i,j} of (indicator(labels[i]==labels[u]) - indicator(labels[i]!=labels[u]))
    #          + sum over u not in {i,j} of (indicator(labels[j]==labels[u]) - indicator(labels[j]!=labels[u]))
    # But we must exclude the edge (i,j) from both sums, and the edge (i,j) itself
    # was crossing before and after, so it contributes 0 to the gain.
    # Let's compute directly:
    # Before: cut contribution from edges incident to i or j:
    #   For each u: edge (i,u) contributes 1 if labels[i]!=labels[u]
    #               edge (j,u) contributes 1 if labels[j]!=labels[u]
    #   Edge (i,j) contributes 1 (since labels differ)
    # After flip:
    #   For each u: edge (i,u) contributes 1 if (1-labels[i])!=labels[u] i.e. labels[i]==labels[u]
    #               edge (j,u) contributes 1 if (1-labels[j])!=labels[u] i.e. labels[j]==labels[u]
    #   Edge (i,j) contributes 1 (still differs)
    # Gain = [sum_u (labels[i]==labels[u]) + sum_u (labels[j]==labels[u])] - [sum_u (labels[i]!=labels[u]) + sum_u (labels[j]!=labels[u])]
    #      - 1 + 1  (the edge (i,j) cancels)
    # Actually, let's just compute the total cut before and after for the affected edges.
    # Simpler: compute gain for flipping i alone + gain for flipping j alone - 2*adj[i,j]
    # Because when we flip i, the edge (i,j) changes status. When we flip j, the edge (i,j) changes again.
    # Flipping i alone: gain_i = same_i - opp_i
    # Flipping j alone: gain_j = same_j - opp_j
    # But when flipping both, the edge (i,j) is counted in both gains.
    # Before: edge (i,j) is crossing (contributes 1)
    # After flipping i: edge (i,j) is not crossing (contributes 0) -> part of gain_i
    # After flipping j (from original): edge (i,j) is not crossing (contributes 0) -> part of gain_j
    # But we flip both simultaneously. The net effect on edge (i,j) is: was crossing, now not crossing? 
    # Wait: labels[i]=0, labels[j]=1. After flip: labels[i]=1, labels[j]=0. Still crossing!
    # So edge (i,j) contribution is unchanged.
    # The gain from flipping i alone includes the change for edge (i,j): 
    #   If labels[i]=0, labels[j]=1: before crossing, after flip i: labels[i]=1, labels[j]=1 -> not crossing. Gain +1 for this edge.
    # The gain from flipping j alone includes the change for edge (i,j):
    #   If labels[i]=0, labels[j]=1: before crossing, after flip j: labels[i]=0, labels[j]=0 -> not crossing. Gain +1 for this edge.
    # But when flipping both: before crossing, after both: labels[i]=1, labels[j]=0 -> still crossing. Gain 0 for this edge.
    # So the combined gain = gain_i + gain_j - 2 * (contribution of edge (i,j) in individual gains)
    # In individual gains, edge (i,j) contributed +1 each (since it was crossing and became not crossing).
    # In combined, it contributes 0. So we subtract 2*1 = 2 if adj[i,j]=1.
    gain_i = _flip_gain(adj, labels, i)
    gain_j = _flip_gain(adj, labels, j)
    # The edge (i,j) is counted in both gain_i and gain_j.
    # In gain_i: if labels[i]!=labels[j], flipping i makes them same, so edge goes from crossing to not: +1
    # In gain_j: if labels[i]!=labels[j], flipping j makes them same, so edge goes from crossing to not: +1
    # In combined: edge stays crossing, so 0.
    # So we need to subtract 2*adj[i,j] from the sum.
    return gain_i + gain_j - 2 * adj[i, j]


def _local_search_1flip(adj, labels, max_time=None, start_time=None):
    """Greedy 1-flip local search."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if max_time is not None and time.time() > max_time:
                return labels
            gain = _flip_gain(adj, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _local_search_2swap(adj, labels, max_time=None, start_time=None):
    """2-swap local search: flip pairs of vertices with different labels."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if max_time is not None and time.time() > max_time:
                return labels
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                gain = _two_swap_gain(adj, labels, i, j)
                if gain > 0:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    improved = True
                    break  # Restart inner loop since labels changed
            # If we broke out of inner loop, outer loop continues
    return labels


def _full_local_search(adj, labels, max_time=None):
    """Alternate between 1-flip and 2-swap local search."""
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    prev_cut = -1
    current_cut = _cut_size(adj, labels)
    for _ in range(50):  # Max iterations
        if time.time() > max_time:
            break
        # 1-flip
        labels = _local_search_1flip(adj, labels, max_time)
        new_cut = _cut_size(adj, labels)
        if new_cut <= current_cut:
            # No improvement, try 2-swap
            labels = _local_search_2swap(adj, labels, max_time)
            new_cut = _cut_size(adj, labels)
            if new_cut <= current_cut:
                break
        current_cut = new_cut
    return labels


def _tabu_search(adj, labels, tabu_len=7, max_iters=500, max_time=None):
    """Cerny tabu search for maximum cut."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    current_cut = best_cut
    
    # Tabu list: stores (vertex, iteration) of last flip
    tabu_until = np.zeros(n, dtype=int)
    current_iter = 0
    
    for it in range(max_iters):
        if time.time() > max_time:
            break
        current_iter = it
        
        # Find the best move (considering tabu restrictions)
        best_gain = -np.inf
        best_v = -1
        best_is_tabu = False
        
        for v in range(n):
            gain = _flip_gain(adj, labels, v)
            is_tabu = (it < tabu_until[v])
            
            # Aspiration criterion: accept tabu move if it improves best
            if is_tabu and gain > best_cut:
                is_tabu = False
            
            if is_tabu:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_v = v
                best_is_tabu = False
        
        if best_v == -1:
            # All moves are tabu, pick the best one anyway
            for v in range(n):
                gain = _flip_gain(adj, labels, v)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1:
                break
        
        # Perform the flip
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = current_iter + tabu_len
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(int)
    except:
        labels = np.arange(n) % 2
    return labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple restarts with different initializations
    # Use tabu search + local search hybrid
    
    inits = []
    
    # Spectral
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Random inits
    np.random.seed(42)
    for _ in range(10):
        rand_labels = np.random.randint(0, 2, n)
        inits.append(rand_labels)
    
    for init_idx, init_labels in enumerate(inits):
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # First, do local search to get to a local optimum
        labels = _full_local_search(adj, labels, max_time=deadline)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Then, apply tabu search starting from this local optimum
        if time.time() > deadline:
            break
        
        ts_labels, ts_cut = _tabu_search(adj, labels.copy(), tabu_len=7, max_iters=300, max_time=deadline)
        
        # Do a final local search on the tabu result
        ts_labels = _full_local_search(adj, ts_labels, max_time=deadline)
        ts_cut = _cut_size(adj, ts_labels)
        
        if ts_cut > best_cut:
            best_cut = ts_cut
            best_labels = ts_labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
