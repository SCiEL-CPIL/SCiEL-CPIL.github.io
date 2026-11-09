import numpy as np
import time


def _spectral_init(adjacency, n):
    """Initialize labels using the Fiedler vector of the graph Laplacian."""
    degrees = adjacency.sum(axis=1).astype(float)
    L = np.diag(degrees) - adjacency.astype(float)
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(L, subset_by_index=[1, 1])
        v = eigvecs[:, 0]
        return (v >= 0).astype(np.int8)
    except Exception:
        return np.random.randint(0, 2, n).astype(np.int8)


def _sa(adjacency, n, labels, row_sums, budget):
    """Run simulated annealing from initial labels. Returns best labels found."""
    labels = labels.copy()
    best_labels = labels.copy()
    # Cut size = 0.5 * sum over v of (row_sums[v] - sum of labels of neighbors)
    # = 0.5 * (2*E - sum_v row_sums[v] + sum_v (A @ labels)[v])
    # Simpler: cut = sum over edges where endpoints differ
    # = 0.5 * sum_v (row_sums[v] - (A @ labels)[v] * 2 - row_sums[v] + ...) 
    # Actually: cut = 0.5 * sum_v (row_sums[v] - (A @ labels)[v])
    # because for each edge (u,v): if same side, contributes 0 to cut; 
    # (A@labels)[v] = sum of labels of neighbors. row_sums[v] - (A@labels)[v] = number of neighbors with label 0 = "other side count" if labels[v]=1, or "same side count" if labels[v]=0.
    # Let me just compute cut directly:
    # cut = 0.5 * (sum of row_sums - sum of (A @ labels) * 2 + ...) 
    # Easiest: cut = np.sum(adjacency * (labels[:, None] != labels[None, :])) / 2
    # But that's O(n^2). Let's use: cut = 0.5 * sum_v (row_sums[v] - (A @ labels)[v])
    # Verify: for vertex v, row_sums[v] - (A @ labels)[v] = # neighbors not labeled same as v... 
    # No: (A @ labels)[v] = sum_{u} A[v,u] * labels[u]. If labels are 0/1, this is the count of neighbors with label 1.
    # If labels[v] = 1: same-side neighbors = (A@labels)[v] - (1 if self-loop else 0). For no self-loops: same = (A@labels)[v], other = row_sums[v] - (A@labels)[v].
    # If labels[v] = 0: same-side neighbors = row_sums[v] - (A@labels)[v], other = (A@labels)[v].
    # So "other" (crossing) = |row_sums[v] - 2*(A@labels)[v]| / ... no.
    # Let's define: crossing_v = number of edges from v to opposite side.
    # If labels[v]=1: crossing_v = row_sums[v] - (A@labels)[v]
    # If labels[v]=0: crossing_v = (A@labels)[v]
    # crossing_v = (A@labels)[v] * (1 - labels[v]) + (row_sums[v] - (A@labels)[v]) * labels[v]
    # = (A@labels)[v] - labels[v]*(A@labels)[v] + row_sums[v]*labels[v] - labels[v]*(A@labels)[v]
    # = (A@labels)[v] - 2*labels[v]*(A@labels)[v] + row_sums[v]*labels[v]
    # Total cut = 0.5 * sum_v crossing_v
    # = 0.5 * sum_v [(A@labels)[v] - 2*labels[v]*(A@labels)[v] + row_sums[v]*labels[v]]
    # = 0.5 * [sum(A@labels) - 2*labels@(A@labels) + row_sums@labels]
    # sum(A@labels) = sum of all entries of A@labels = sum of row_sums * labels? No.
    # sum_v (A@labels)[v] = sum_v sum_u A[v,u]*labels[u] = sum_u labels[u] * sum_v A[v,u] = sum_u labels[u]*row_sums[u] = row_sums @ labels
    # So cut = 0.5 * [row_sums@labels - 2*labels@(A@labels) + row_sums@labels]
    # = 0.5 * [2*(row_sums@labels) - 2*labels@(A@labels)]
    # = (row_sums@labels) - labels@(A@labels)
    # = labels @ row_sums - labels @ A @ labels
    # = labels @ (row_sums - A @ labels)
    
    al = adjacency @ labels
    cut = np.dot(labels, row_sums - al)
    best_cut = cut
    
    t = 1.0
    t_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    while t > t_min:
        if time.time() - start_time > budget:
            break
        
        # Pick a random vertex
        v = rng.integers(n)
        
        # Compute delta if we flip v
        # al[v] = sum of labels of neighbors of v
        # If labels[v] = 1: same = al[v], other = row_sums[v] - al[v]. Flip: new same = row_sums[v]-al[v], new other = al[v]. Delta = 2*(al[v] - (row_sums[v]-al[v])) = 4*al[v] - 2*row_sums[v]
        # If labels[v] = 0: same = row_sums[v]-al[v], other = al[v]. Flip: new same = al[v], new other = row_sums[v]-al[v]. Delta = 2*((row_sums[v]-al[v]) - al[v]) = 2*row_sums[v] - 4*al[v]
        # In both cases: delta = 2 * (other - same) where other/same depend on current label
        # delta = 2 * (other - same)
        # If labels[v]=1: other = row_sums[v]-al[v], same = al[v]. delta = 2*(row_sums[v]-2*al[v])
        # If labels[v]=0: other = al[v], same = row_sums[v]-al[v]. delta = 2*(2*al[v]-row_sums[v])
        # Unified: delta = 2 * (2*al[v] - row_sums[v]) * (1 if labels[v]==0 else -1)
        # = 2 * (2*al[v] - row_sums[v]) * (1 - 2*labels[v])
        
        lv = labels[v]
        delta = 2.0 * (2.0 * al[v] - row_sums[v]) * (1.0 - 2.0 * lv)
        
        if delta >= 0 or rng.random() < np.exp(delta / t):
            labels[v] = 1 - lv
            # Update al: flipping v changes al[u] for all neighbors u of v by +/-1
            # If v was 1 and becomes 0: al[u] -= 1 for neighbors u
            # If v was 0 and becomes 1: al[u] += 1 for neighbors u
            if lv == 1:
                al[adjacency[v] == 1] -= 1
            else:
                al[adjacency[v] == 1] += 1
            cut += delta
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        t *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.int8)
    n = len(adjacency)
    row_sums = adjacency.sum(axis=1).astype(float)
    
    deadline = 110.0  # safety margin from 120s
    start_time = time.time()
    
    best_labels = None
    best_cut = -1
    
    # Try spectral init first
    spec_labels = _spectral_init(adjacency, n)
    al = adjacency @ spec_labels
    cut = np.dot(spec_labels, row_sums - al)
    if cut > best_cut:
        best_cut = cut
        best_labels = spec_labels.copy()
    
    # Run SA from spectral init with remaining time
    elapsed = time.time() - start_time
    if elapsed < deadline:
        budget = deadline - elapsed - 2.0  # 2s safety
        if budget > 5.0:
            labels, cut = _sa(adjacency, n, spec_labels, row_sums, budget)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    # If time remains, try a few random restarts
    elapsed = time.time() - start_time
    rng = np.random.default_rng(123)
    for restart in range(3):
        if time.time() - start_time > deadline - 5.0:
            break
        rand_labels = rng.integers(0, 2, n).astype(np.int8)
        al = adjacency @ rand_labels
        cut = np.dot(rand_labels, row_sums - al)
        if cut > best_cut:
            best_cut = cut
            best_labels = rand_labels.copy()
        
        elapsed = time.time() - start_time
        budget = deadline - elapsed - 2.0
        if budget > 3.0:
            labels, cut = _sa(adjacency, n, rand_labels, row_sums, budget)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    # Final greedy polish on the best solution
    if best_labels is not None and time.time() - start_time < deadline - 1.0:
        labels = best_labels.copy()
        for _ in range(2):
            for v in range(n):
                same = adjacency[v][labels == labels[v]].sum()
                other = adjacency[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
        al = adjacency @ labels
        cut = np.dot(labels, row_sums - al)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels
