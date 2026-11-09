<<<<<<< SEARCH
def _nj(dmat, labels):
    """Neighbor Joining algorithm on a distance matrix."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dmat.copy()
    
    while len(active) > 3:
        k = len(active)
        r = np.array([sum(d[i][j] for j in active if j != i) for i in active])
        
        best_q = float('inf')
        best_i = best_j = None
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, len(active)):
                j = active[idx_j]
                q = (k - 2) * d[i][j] - r[idx_i] - r[idx_j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        new_idx = k
        trees[new_idx] = f"({trees[i]},{trees[j]})"
        
        for x in active:
            if x != i and x != j:
                d[new_idx][x] = 0.5 * (d[i][x] + d[j][x] - d[i][j])
                d[x][new_idx] = d[new_idx][x]
        
        active = [x for x in active if x != i and x != j]
        active.append(new_idx)
    
    a, b, c = active
    return f"({trees[a]},{trees[b]},{trees[c]})"
=======
def _nj_numpy_fast(dmat, labels):
    """Vectorized Neighbor Joining algorithm."""
    n = len(labels)
    if n <= 3:
        if n == 1: return labels[0]
        if n == 2: return f"({labels[0]},{labels[1]})"
        return f"({labels[0]},{labels[1]},{labels[2]})"
        
    max_nodes = 2 * n - 2
    d = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    d[:n, :n] = dmat
    
    active = list(range(n))
    topology = {i: labels[i] for i in range(n)}
    next_node = n
    current_n = n
    
    while current_n > 3:
        k = current_n
        active_arr = np.array(active)
        sub_d = d[np.ix_(active_arr, active_arr)]
        
        r = sub_d.sum(axis=1)
        Q = (k - 2) * sub_d - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        i_node = active[i_idx]
        j_node = active[j_idx]
        
        new_node = next_node
        next_node += 1
        topology[new_node] = (i_node, j_node)
        
        mask = np.ones(k, dtype=bool)
        mask[i_idx] = False
        mask[j_idx] = False
        x_indices = np.where(mask)[0]
        
        if len(x_indices) > 0:
            x_nodes = active_arr[x_indices]
            new_dists = 0.5 * (sub_d[i_idx, x_indices] + sub_d[j_idx, x_indices] - sub_d[i_idx, j_idx])
            d[new_node, x_nodes] = new_dists
            d[x_nodes, new_node] = new_dists
            
        active.remove(i_node)
        active.remove(j_node)
        active.append(new_node)
        current_n -= 1
        
    a, b, c = active
    return f"({topology[a]},{topology[b]},{topology[c]})"
>>>>>>> REPLACE

