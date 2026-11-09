<<<<<<< SEARCH
def _simulated_annealing(adjacency, labels, t0, tf, alpha, max_iter, deadline):
    """Simulated annealing to escape local optima."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    
    current_labels = labels.copy()
    current_cut = _cut_size(adjacency, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        new_labels = current_labels.copy()
        new_labels[v] = 1 - new_labels[v]
        
        # Compute new cut size
        new_cut = _cut_size(adjacency, new_labels)
        delta = new_cut - current_cut
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta > 0:
            current_labels = new_labels
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = new_labels.copy()
        else:
            T = t0 * (alpha ** it)
            if T > 1e-10:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_labels = new_labels
                    current_cut = new_cut
    
    return best_labels, best_cut
=======
def _walksat(adjacency, labels, p, max_iter, deadline):
    """WalkSAT-style stochastic local search for Max-2-SAT.
    
    For each edge (u,v) that is "uncut" (both endpoints same label), 
    we have an unsatisfied clause. We pick a random uncut edge, then flip
    one of its endpoints with probability proportional to how many other
    uncut edges that vertex is involved in (biased toward fixing more clauses).
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    
    current_labels = labels.copy()
    current_cut = _cut_size(adjacency, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    # Precompute adjacency lists for faster neighbor iteration
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find all uncut edges (edges within same partition)
        # An edge (u,v) is uncut if labels[u] == labels[v]
        # We need to pick one such edge to "repair"
        
        # Efficient approach: for each vertex, count how many neighbors are in same set
        # Then pick a vertex with positive same-set degree, weighted by that count
        same_set_degrees = np.zeros(n)
        for v in range(n):
            if same_set_degrees[v] == 0 and len(neighbors[v]) > 0:
                # Count neighbors in same partition
                nbrs = neighbors[v]
                if len(nbrs) > 0:
                    same_set_degrees[v] = (current_labels[nbrs] == current_labels[v]).sum()
        
        # Find vertices with uncut edges
        candidates = np.where(same_set_degrees > 0)[0]
        if len(candidates) == 0:
            # No uncut edges - we're at a local optimum (or global)
            # Do a random flip to escape
            v = np.random.randint(n)
        else:
            # Weighted selection: prefer vertices with more uncut edges
            weights = same_set_degrees[candidates]
            # Add small random component for exploration
            weights = weights + 0.1 * np.random.rand(len(weights))
            v = candidates[np.random.choice(len(weights), p=weights/weights.sum())]
        
        # Flip vertex v
        current_labels[v] = 1 - current_labels[v]
        new_cut = _cut_size(adjacency, current_labels)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = current_labels.copy()
        
        # With probability (1-p), do a random flip instead (exploration)
        if np.random.random() > p:
            v_random = np.random.randint(n)
            current_labels[v_random] = 1 - current_labels[v_random]
    
    return best_labels, best_cut
>>>>>>> REPLACE

