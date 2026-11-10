"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import itertools

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    # Using a lookup table for speed
    enc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Manual loop might be faster for short strings than frombuffer if not aligned?
        # But frombuffer is very fast. Let's stick to frombuffer for consistency with seed.
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        # Map ASCII to 0-3
        # A=65, C=67, G=71, T=84
        # We can do a simple lookup or arithmetic.
        # 65->0, 67->1, 71->2, 84->3
        # (65, 67, 71, 84)
        # Let's just use a lookup array for safety
        lookup = np.zeros(256, dtype=np.int8)
        lookup[65] = 0 # A
        lookup[67] = 1 # C
        lookup[71] = 2 # G
        lookup[84] = 3 # T
        enc[i] = lookup[arr]
    
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs
    
    p = diff_matrix.astype(np.float64) / seq_len
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels):
    """
    Perform neighbor joining on a distance matrix D.
    Returns a Newick string (no trailing semicolon).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)
    
    while len(node_names) > 3:
        m = len(node_names)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        new_name = f"({node_names[i]},{node_names[j]})"
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1
        
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0
        new_names = [node_names[k] for k in keep] + [new_name]
        D = new_D
        node_names = new_names

    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _parse_newick_to_tree(newick, labels):
    """
    Parse a Newick string into an unrooted tree represented as adjacency.
    Returns:
        adj: dict {node_id: set(neighbor_ids)}
        leaf_to_node: dict {label: node_id}
        node_to_label: dict {node_id: label}
        root_id: int
    """
    newick = newick.strip().rstrip(';')
    nodes = []
    adj = {}
    leaf_to_node = {}
    node_to_label = {}
    
    node_id_counter = [0]
    
    def next_node_id():
        nid = node_id_counter[0]
        node_id_counter[0] += 1
        nodes.append(nid)
        adj[nid] = set()
        return nid
    
    def parse(s, pos):
        if pos >= len(s):
            raise ValueError("Unexpected end of Newick")
        
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child_id, pos = parse(s, pos)
                children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    raise ValueError(f"Expected , or ) at pos {pos}")
            
            parent_id = next_node_id()
            for child_id in children:
                adj[parent_id].add(child_id)
                adj[child_id].add(parent_id)
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            
            return parent_id, pos
        else:
            name_end = pos
            while name_end < len(s) and s[name_end] not in '():,':
                name_end += 1
            name = s[pos:name_end]
            leaf_id = next_node_id()
            leaf_to_node[name] = leaf_id
            node_to_label[leaf_id] = name
            pos = name_end
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            
            return leaf_id, pos
    
    root_id, _ = parse(newick, 0)
    
    return adj, leaf_to_node, node_to_label, root_id


def _fit_branch_lengths_least_squares(adj, leaf_to_node, node_to_label, D_obs):
    """
    Fit branch lengths to the tree topology to minimize SSE against D_obs.
    Returns the SSE.
    """
    n_leaves = len(leaf_to_node)
    n_nodes = len(adj)
    
    # Get list of leaves
    leaf_ids = list(leaf_to_node.values())
    
    # We need to set up the linear system for branch lengths.
    # For each pair of leaves (i, j), the distance is the sum of branch lengths on the path.
    # This is a system of linear equations: A * x = b
    # Where x is the vector of branch lengths (length n_nodes - 1 for unrooted tree? No, n_nodes - 1 edges).
    # Actually, for an unrooted tree with L leaves, there are 2*L - 2 nodes and 2*L - 3 edges.
    # The system is underdetermined or just determined?
    # Number of equations: L*(L-1)/2.
    # Number of unknowns: 2*L - 3.
    # For L=4, 6 equations, 5 unknowns. Overdetermined.
    
    # We use Least Squares to find x.
    # We need to construct the incidence matrix A.
    
    # First, get all edges
    edges = []
    edge_map = {} # (u, v) -> edge_idx
    for u in adj:
        for v in adj[u]:
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
                edge_map[(v, u)] = len(edges)
    
    n_edges = len(edges)
    
    # Construct A (n_pairs x n_edges) and b (n_pairs)
    # Only use pairs of leaves
    leaf_pairs = list(itertools.combinations(leaf_ids, 2))
    n_pairs = len(leaf_pairs)
    
    A = np.zeros((n_pairs, n_edges))
    b = np.zeros(n_pairs)
    
    # For each pair, find the path in the tree and mark the edges
    # BFS/DFS from leaf i to leaf j
    
    # Pre-compute paths? Might be slow for large N.
    # For N=32, n_pairs = 496. n_edges = 61.
    # 496 * (N log N) is fine.
    
    for idx, (li, lj) in enumerate(leaf_pairs):
        # BFS from li to lj
        visited = {li: None}
        queue = [li]
        path_edges = []
        
        while queue:
            u = queue.pop(0)
            if u == lj:
                break
            for v in adj[u]:
                if v not in visited:
                    visited[v] = u
                    queue.append(v)
        
        # Reconstruct path
        if lj not in visited:
            continue # Should not happen
        
        curr = lj
        while curr != li:
            parent = visited[curr]
            # Edge (parent, curr)
            edge_idx = edge_map.get((parent, curr))
            if edge_idx is not None:
                A[idx, edge_idx] = 1.0
            curr = parent
        
        # Get observed distance
        # Map leaf IDs to labels to index D_obs
        lab_i = node_to_label[li]
        lab_j = node_to_label[lj]
        # Find indices in D_obs
        # D_obs is indexed by labels sorted in reconstruct?
        # We need to pass the labels list to this function or ensure D_obs is aligned with node_to_label keys.
        # In reconstruct, we will pass D_obs aligned with the labels used in NJ.
        
        # We need a mapping from label to index in D_obs
        # This function doesn't have the labels list.
        # Let's assume D_obs is a dict or we pass the labels.
        # Better: Pass labels list.
        
    # This implementation is getting messy with indices.
    # Let's simplify: 
    # We will compute the SSE by:
    # 1. Solving for branch lengths x using np.linalg.lstsq.
    # 2. Computing implied distances D_tree.
    # 3. Calculating SSE.
    
    # To make it robust, let's pass the labels list.
    
    return -1.0 # Placeholder


def _evaluate_tree_sse(newick, labels, D_obs):
    """
    Evaluate the SSE of a tree given the distance matrix.
    """
    adj, leaf_to_node, node_to_label, root_id = _parse_newick_to_tree(newick, labels)
    
    n_leaves = len(leaf_to_node)
    if n_leaves < 2:
        return 0.0
    
    # Map label to index in D_obs
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Get edges
    edges = []
    edge_map = {}
    for u in adj:
        for v in adj[u]:
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
                edge_map[(v, u)] = len(edges)
    
    n_edges = len(edges)
    
    # Get leaf pairs
    leaf_ids = list(leaf_to_node.values())
    leaf_pairs = list(itertools.combinations(leaf_ids, 2))
    n_pairs = len(leaf_pairs)
    
    A = np.zeros((n_pairs, n_edges))
    b = np.zeros(n_pairs)
    
    for idx, (li, lj) in enumerate(leaf_pairs):
        # BFS
        visited = {li: None}
        queue = [li]
        
        while queue:
            u = queue.pop(0)
            if u == lj:
                break
            for v in adj[u]:
                if v not in visited:
                    visited[v] = u
                    queue.append(v)
        
        if lj not in visited:
            continue
            
        curr = lj
        while curr != li:
            parent = visited[curr]
            edge_idx = edge_map.get((parent, curr))
            if edge_idx is not None:
                A[idx, edge_idx] = 1.0
            curr = parent
        
        lab_i = node_to_label[li]
        lab_j = node_to_label[lj]
        b[idx] = D_obs[label_to_idx[lab_i], label_to_idx[lab_j]]
    
    # Solve least squares
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # Ensure non-negative branch lengths? 
        # Least squares might give negative lengths. 
        # We can clip them to 0 and re-calculate, or just use the SSE from the unclipped solution as a proxy.
        # Clipping is better for validity.
        x_clipped = np.maximum(x, 0.0)
        
        # Calculate implied distances
        D_tree = A @ x_clipped
        SSE = np.sum((b - D_tree) ** 2)
        
        return SSE
    except:
        return float('inf')


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Generate candidate trees
    candidates = []
    
    # 1. Standard NJ
    tree_1 = _neighbor_joining(D, labels)
    candidates.append(tree_1)
    
    # 2. Perturbed NJ runs
    # Use a fixed seed for reproducibility in case we need it, but random is fine
    rng = np.random.default_rng(42)
    
    num_perturbations = 5 # Keep low to save time
    for i in range(num_perturbations):
        if time.time() > deadline - 10:
            break
        
        # Add small noise
        noise = rng.normal(0, 0.01, D.shape)
        D_noisy = D + noise
        # Ensure symmetry
        D_noisy = (D_noisy + D_noisy.T) * 0.5
        np.fill_diagonal(D_noisy, 0.0)
        
        tree_p = _neighbor_joining(D_noisy, labels)
        candidates.append(tree_p)
    
    # Evaluate candidates
    best_tree = candidates[0]
    best_sse = float('inf')
    
    for tree in candidates:
        if time.time() > deadline - 5:
            break
        
        sse = _evaluate_tree_sse(tree, labels, D)
        if sse < best_sse:
            best_sse = sse
            best_tree = tree
    
    return best_tree + ";"
# EVOLVE-BLOCK-END
