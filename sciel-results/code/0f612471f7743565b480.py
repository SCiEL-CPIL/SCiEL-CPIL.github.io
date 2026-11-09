<<<<<<< SEARCH
def _nj_build(labels, dist_mat):
    """
    Neighbor-Joining tree reconstruction.
    Returns a Newick string (unrooted).
    labels: list of taxon labels
    dist_mat: n x n distance matrix
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Work with indices
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # dist is indexed by original indices; we'll maintain a dict for current distances
    # Use a mutable structure: dist_dict[(i,j)] for current active nodes
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = dist_mat[i][j]
                dist[(j, i)] = dist_mat[i][j]
    
    counter = n  # for naming new internal nodes
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        r = {a: 0.0 for a in active}
        for a in active:
            for b in active:
                if a != b:
                    r[a] += dist[(a, b)]
        
        # Find best pair
        best_c = float('inf')
        best_a, best_b = active[0], active[1]
        for i_idx, a in enumerate(active):
            for b in active[i_idx + 1:]:
                c = (m - 2) * dist[(a, b)] - r[a] - r[b]
                if c < best_c:
                    best_c = c
                    best_a, best_b = a, b
        
        # Create new node
        new_id = counter
        counter += 1
        trees[new_id] = f"({trees[best_a]},{trees[best_b]})"
        
        # Compute distances from new node to all other active nodes
        for x in active:
            if x not in (best_a, best_b):
                d = 0.5 * (dist[(best_a, x)] + dist[(best_b, x)] - dist[(best_a, best_b)])
                dist[(new_id, x)] = d
                dist[(x, new_id)] = d
        
        # Remove old nodes, add new
        active = [x for x in active if x not in (best_a, best_b)]
        active.append(new_id)
    
    # Combine remaining 3 nodes
    a, b, c = active
    return f"({trees[a]},{trees[b]},{trees[c]});"
=======
def _nj_build(labels, dist_mat):
    """
    Neighbor-Joining tree reconstruction for initial topology.
    Returns a list of (child1, child2) for internal nodes, with leaves as strings.
    We represent the tree as a list where each element is either:
    - A string (leaf label)
    - A tuple (child1, child2) for internal nodes
    """
    n = len(labels)
    if n <= 1:
        return [labels[0]] if n == 1 else []
    if n == 2:
        return [(labels[0], labels[1])]
    
    # Work with indices
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = dist_mat[i][j]
                dist[(j, i)] = dist_mat[i][j]
    
    counter = n
    while len(active) > 3:
        m = len(active)
        r = {a: 0.0 for a in active}
        for a in active:
            for b in active:
                if a != b:
                    r[a] += dist[(a, b)]
        
        best_c = float('inf')
        best_a, best_b = active[0], active[1]
        for i_idx, a in enumerate(active):
            for b in active[i_idx + 1:]:
                c = (m - 2) * dist[(a, b)] - r[a] - r[b]
                if c < best_c:
                    best_c = c
                    best_a, best_b = a, b
        
        new_id = counter
        counter += 1
        trees[new_id] = (trees[best_a], trees[best_b])
        
        for x in active:
            if x not in (best_a, best_b):
                d = 0.5 * (dist[(best_a, x)] + dist[(best_b, x)] - dist[(best_a, best_b)])
                dist[(new_id, x)] = d
                dist[(x, new_id)] = d
        
        active = [x for x in active if x not in (best_a, best_b)]
        active.append(new_id)
    
    a, b, c = active
    return [trees[a], trees[b], trees[c]]


def _compute_jc_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix using numpy for speed."""
    arr = _encode_sequences(sequences, labels)
    n = len(labels)
    mat = np.zeros((n, n))
    n_sites = arr.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(arr[i] != arr[j])
            p = diff / n_sites
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat
>>>>>>> REPLACE

