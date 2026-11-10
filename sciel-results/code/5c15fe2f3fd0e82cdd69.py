import numpy as np
import time
import math
import sys

def _seq_to_int(sequences, labels):
    """Convert sequences to int8 array."""
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(labels)
    L = len(sequences[labels[0]])
    seq_int = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j in range(L):
            c = s[j]
            seq_int[i, j] = char_map.get(c, 0)
    return seq_int, L

def _jc_distance_matrix(seq_int, L):
    """Compute JC-corrected pairwise distances efficiently."""
    n = seq_int.shape[0]
    # Use bit-packing for efficiency
    # For each nucleotide, create a binary matrix (n, L)
    # mismatch(i,j) = L - sum_s (A_s[i] & A_s[j]).sum()
    # where A_s is the indicator that position has nucleotide s
    
    # Build indicator matrices
    indicators = []
    for s in range(4):
        ind = (seq_int == s).astype(np.int32)  # (n, L)
        indicators.append(ind)
    
    match_count = np.zeros((n, n), dtype=np.int32)
    for ind in indicators:
        match_count += ind @ ind.T
    
    mismatch = L - match_count
    np.fill_diagonal(mismatch, 0)
    
    p = mismatch.astype(np.float64) / L
    p = np.clip(p, 1e-10, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    return d

def _nj_build(d):
    """Neighbor-Joining algorithm. Returns tree structure.
    Returns: list of tuples (node, parent) or similar structure for Newick generation.
    We'll use a dictionary-based approach for clarity and correctness.
    """
    n = d.shape[0]
    # Nodes: 0..n-1 are leaves, n.. are internal
    # dist: dictionary mapping frozenset of node pairs or just maintain a matrix
    # For efficiency with up to 32 taxa, a matrix is fine
    
    # Active nodes
    active = list(range(n))
    # Distance matrix for active nodes (will be rebuilt each iteration)
    # Actually, let's maintain a full matrix and mark inactive
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = d
    
    # Tree structure: for each node, store its children
    children = {i: [] for i in range(max_nodes)}
    parent = {i: -1 for i in range(max_nodes)}
    is_leaf = {i: True for i in range(max_nodes)}
    next_idx = n
    
    while len(active) > 2:
        k = len(active)
        idx = np.array(active)
        sub_d = D[np.ix_(idx, idx)]
        r = sub_d.sum(axis=1)
        Q = (k - 2) * sub_d - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        u = active[i_idx]
        v = active[j_idx]
        w = next_idx
        next_idx += 1
        
        # Update distances
        for x in active:
            if x == u or x == v:
                continue
            D[w, x] = 0.5 * (D[u, x] + D[v, x] - D[u, v])
            D[x, w] = D[w, x]
        D[w, w] = 0.0
        
        # Update tree structure
        active = [x for x in active if x != u and x != v]
        active.append(w)
        
        children[u] = []
        children[v] = []
        children[w] = [u, v]
        parent[u] = w
        parent[v] = w
        is_leaf[w] = False
        
        # Mark u and v as no longer active in the "top-level" sense
        # But they remain as nodes in the tree
    
    # Connect the last two nodes
    u, v = active[0], active[1]
    w = next_idx
    next_idx += 1
    children[w] = [u, v]
    parent[u] = w
    parent[v] = w
    is_leaf[w] = False
    
    root = w
    return root, children, parent, is_leaf, next_idx

def _tree_to_newick(root, children, is_leaf):
    """Convert tree to Newick string."""
    def to_newick(node):
        if is_leaf[node]:
            return f"t{node}"
        ch = children[node]
        if len(ch) == 2:
            return f"({to_newick(ch[0])},{to_newick(ch[1])})"
        elif len(ch) == 1:
            return to_newick(ch[0])
        else:
            return f"t{node}"
    return to_newick(root) + ";"

def _get_splits(root, children, is_leaf, n):
    """Get all non-trivial splits of the tree."""
    splits = set()
    
    def get_split(node, parent_node):
        if is_leaf[node]:
            return frozenset([node])
        
        leaf_set = set()
        for ch in children[node]:
            leaf_set.update(get_split(ch, node))
        
        if 0 < len(leaf_set) < n:
            # Normalize: always have 0 in the first set
            if 0 in leaf_set:
                key = (frozenset(leaf_set), frozenset(set(range(n)) - leaf_set))
            else:
                key = (frozenset(set(range(n)) - leaf_set), frozenset(leaf_set))
            splits.add(key)
        
        return leaf_set
    
    get_split(root, -1)
    return splits

def _robinson_foulds_distance(splits1, splits2, n):
    """Compute normalized Robinson-Foulds distance."""
    # Count common splits
    common = 0
    for s in splits1:
        if s in splits2:
            common += 1
    
    total1 = len(splits1)
    total2 = len(splits2)
    
    # Normalized RF distance
    if total1 + total2 == 0:
        return 0.0
    return (total1 + total2 - 2 * common) / (total1 + total2)

def _jc_probs(bl):
    """Return (p_same, p_diff) for JC branch length bl."""
    if bl < 0:
        bl = 0.0
    e = math.exp(-4.0 * bl / 3.0)
    return 0.25 + 0.75 * e, 0.25 * (1.0 - e)

def _compute_ll(root, children, is_leaf, n, seq_int, L, branch_len):
    """Felsenstein pruning. Returns total log-likelihood.
    branch_len[node] = length of branch from parent to node."""
    # Build postorder traversal
    postorder = []
    def dfs(node):
        if is_leaf[node]:
            postorder.append(node)
        else:
            for ch in children[node]:
                dfs(ch)
            postorder.append(node)
    
    dfs(root)
    
    max_nodes = 2 * n
    ll = np.zeros((max_nodes, 4), dtype=np.float64)
    
    for node in postorder:
        if is_leaf[node]:
            obs = seq_int[node]
            for s in range(4):
                ll[node, s] = 1.0 if s == obs else 0.0
        else:
            ch = children[node]
            if len(ch) == 2:
                c1, c2 = ch
                ps1, pd1 = _jc_probs(branch_len[c1])
                ps2, pd2 = _jc_probs(branch_len[c2])
                
                for s in range(4):
                    total = 0.0
                    for s1 in range(4):
                        p1 = ps1 if s == s1 else pd1
                        v1 = p1 * ll[c1, s1]
                        for s2 in range(4):
                            p2 = ps2 if s == s2 else pd2
                            total += v1 * p2 * ll[c2, s2]
                    ll[node, s] = total
            elif len(ch) == 1:
                c1 = ch[0]
                ps1, pd1 = _jc_probs(branch_len[c1])
                for s in range(4):
                    total = 0.0
                    for s1 in range(4):
                        p1 = ps1 if s == s1 else pd1
                        total += p1 * ll[c1, s1]
                    ll[node, s] = total
    
    # Root: uniform prior
    total = 0.25 * np.sum(ll[root])
    if total <= 0:
        return -1e10
    return math.log(total)

def _nni_refine(root, children, is_leaf, n, seq_int, L, deadline):
    """NNI refinement using JC likelihood."""
    max_nodes = 2 * n
    branch_len = np.zeros(max_nodes, dtype=np.float64)
    
    # Initialize branch lengths
    for i in range(max_nodes):
        if i < n:
            branch_len[i] = 0.05
        else:
            branch_len[i] = 0.03
    
    current_ll = _compute_ll(root, children, is_leaf, n, seq_int, L, branch_len)
    
    # Find all internal edges
    def get_internal_edges():
        edges = []
        def dfs(node, parent_node):
            if not is_leaf[node]:
                for ch in children[node]:
                    if not is_leaf[ch]:
                        edges.append((node, ch))
                    dfs(ch, node)
        dfs(root, -1)
        return edges
    
    improved = True
    max_iterations = 30
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline:
        improved = False
        iteration += 1
        
        internal_edges = get_internal_edges()
        
        for (a, b) in internal_edges:
            if time.time() > deadline:
                break
            
            # Find neighbors of a other than b
            neighbors_a = []
            for ch in children[a]:
                if ch != b:
                    neighbors_a.append(ch)
            # Also check parent
            # In our rooted representation, we need to be careful
            # For now, let's only consider child-internal edges
            
            # Find neighbors of b other than a
            neighbors_b = []
            for ch in children[b]:
                if ch != a:  # b's children, none should be a since a is parent
                    neighbors_b.append(ch)
            
            # For NNI, we need exactly 2 subtrees on each side
            # This is tricky with rooted trees. Let's use a simpler approach:
            # Just try swapping subtrees directly
            
            if len(neighbors_a) < 2 or len(neighbors_b) < 2:
                continue
            
            # Take first two from each
            c, d = neighbors_a[0], neighbors_a[1]
            e, f = neighbors_b[0], neighbors_b[1]
            
            # Save state
            old_children_a = list(children[a])
            old_children_b = list(children[b])
            
            # Try NNI move: swap d and e
            # a gets c, e; b gets d, f
            children[a] = [c, e]
            children[b] = [d, f]
            
            new_ll = _compute_ll(root, children, is_leaf, n, seq_int, L, branch_len)
            
            if new_ll > current_ll + 1e-10:
                current_ll = new_ll
                improved = True
            else:
                # Revert
                children[a] = old_children_a
                children[b] = old_children_b
    
    return root, children, is_leaf, branch_len

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 160.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Convert sequences to integer array
    seq_int, L = _seq_to_int(sequences, labels)
    
    # Compute JC distance matrix
    d = _jc_distance_matrix(seq_int, L)
    
    # Build NJ tree
    root, children, is_leaf, next_idx = _nj_build(d)
    
    # NNI refinement
    try:
        root, children, is_leaf, branch_len = _nni_refine(
            root, children, is_leaf, n, seq_int, L, deadline
        )
    except Exception:
        pass
    
    # Convert to Newick
    newick = _tree_to_newick(root, children, is_leaf)
    return newick
