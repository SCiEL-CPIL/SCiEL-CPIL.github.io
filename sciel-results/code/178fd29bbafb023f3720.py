"""Rewritten program for phylogenetic topology recovery using Maximum Likelihood."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

_DEADLINE = 265.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _encode_sequences(sequences, labels):
    """Encode DNA sequences as integer arrays."""
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(labels)
    s_len = len(sequences[labels[0]])
    seqs = np.zeros((n, s_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Use a lookup table or vectorized mapping
        # For speed, we can use a precomputed map if possible, but simple loop is okay for N<=32
        for j, c in enumerate(s):
            seqs[i, j] = seq_map[c]
    return seqs


def _jc_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix."""
    seqs = _encode_sequences(sequences, labels)
    n = len(labels)
    num_sites = seqs.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(seqs[i] != seqs[j])
            p = diffs / num_sites
            if p >= 0.75:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _neighbor_joining(dist):
    """
    Neighbor Joining algorithm. Returns a list of edges (i, j, length).
    We represent the tree as a dictionary of children for easier manipulation.
    """
    n = len(dist)
    if n <= 2:
        return []
    
    active = list(range(n))
    d = dist.copy()
    # Tree structure: list of edges (node1, node2)
    edges = []
    next_node = n
    
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for i_idx in range(m):
            ai = active[i_idx]
            for j_idx in range(m):
                if i_idx != j_idx:
                    r[i_idx] += d[ai, active[j_idx]]
        
        best_criterion = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        for i_idx in range(m):
            for j_idx in range(i_idx + 1, m):
                ai = active[i_idx]
                aj = active[j_idx]
                criterion = (m - 2) * d[ai, aj] - r[i_idx] - r[j_idx]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i_idx = i_idx
                    best_j_idx = j_idx
        
        if best_i_idx == -1:
            break
        
        ai = active[best_i_idx]
        aj = active[best_j_idx]
        new_node = next_node
        next_node += 1
        
        # Add edges
        edges.append((ai, new_node))
        edges.append((aj, new_node))
        
        # Compute distances
        for k_idx in range(m):
            if k_idx == best_i_idx or k_idx == best_j_idx:
                continue
            ak = active[k_idx]
            d[new_node, ak] = 0.5 * (d[ai, ak] + d[aj, ak] - d[ai, aj])
            d[ak, new_node] = d[new_node, ak]
        
        new_active = []
        for k_idx in range(m):
            if k_idx != best_i_idx and k_idx != best_j_idx:
                new_active.append(active[k_idx])
        new_active.append(new_node)
        active = new_active
        
        if _elapsed() > _DEADLINE - 10:
            break
    
    # Connect remaining 3 nodes
    if len(active) == 3:
        a, b, c = active
        root = next_node
        edges.append((a, root))
        edges.append((b, root))
        edges.append((c, root))
    elif len(active) == 2:
        a, b = active
        root = next_node
        edges.append((a, root))
        edges.append((b, root))
    
    return edges


def _build_tree_from_edges(edges, n_leaves):
    """
    Build a tree structure from edges.
    Returns a dictionary: node -> list of children
    and a list of branch lengths (to be optimized later).
    We need to identify the root. Let's root at the first internal node or a leaf.
    For likelihood, we can root anywhere. Let's root at node 0 (a leaf) for simplicity, 
    or better, root at an internal node to balance the tree.
    """
    # Identify internal nodes
    all_nodes = set()
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    
    # Leaves are 0..n_leaves-1
    leaves = set(range(n_leaves))
    internal_nodes = all_nodes - leaves
    
    # Build adjacency list
    adj = {node: [] for node in all_nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a root. Pick an internal node if possible.
    root = next(iter(internal_nodes)) if internal_nodes else 0
    
    # BFS to determine parent-child relationships
    children = {node: [] for node in all_nodes}
    parent = {node: None for node in all_nodes}
    visited = {root}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                queue.append(v)
                
    return children, root, edges


def _calculate_likelihood(children, root, branch_lengths, seqs):
    """
    Calculate log-likelihood using Felsenstein's pruning algorithm.
    branch_lengths: dict node -> length of branch to parent.
    seqs: (n_leaves, S) array of ints.
    """
    n_leaves = len(seqs)
    S = seqs.shape[1]
    
    # Initialize likelihoods for leaves
    # L_v is an array of shape (S, 4)
    # For a leaf v, L_v[x] = 1 if seqs[v] == x, else 0
    L = {}
    for v in range(n_leaves):
        L[v] = np.zeros((S, 4), dtype=np.float64)
        L[v][np.arange(S), seqs[v]] = 1.0
    
    # Process internal nodes in reverse topological order (post-order)
    # We need a post-order traversal
    def get_post_order(node):
        order = []
        def dfs(u):
            for c in children[u]:
                dfs(c)
            order.append(u)
        dfs(root)
        return order
        
    order = get_post_order(root)
    
    for v in order:
        if v < n_leaves:
            continue # Already set
        
        # Calculate L_v
        # L_v(x) = prod_{w in children(v)} sum_{y} L_w(y) P_{x->y}(t_w)
        # We start with ones
        L_v = np.ones((S, 4), dtype=np.float64)
        
        for w in children[v]:
            t_w = branch_lengths[w]
            # Compute P_{x->y}(t)
            # P_same = 0.25 + 0.75 * exp(-4t/3)
            # P_diff = 0.25 - 0.25 * exp(-4t/3)
            exp_term = math.exp(-4.0 * t_w / 3.0)
            p_same = 0.25 + 0.75 * exp_term
            p_diff = 0.25 - 0.25 * exp_term
            
            # For each site s and state x:
            # sum_y L_w(s, y) P_{x->y}
            # = L_w(s, x) * p_same + sum_{y != x} L_w(s, y) * p_diff
            # = L_w(s, x) * (p_same - p_diff) + p_diff * sum_y L_w(s, y)
            # = L_w(s, x) * exp_term + p_diff * sum_y L_w(s, y)
            
            L_w = L[w]
            sum_L_w = np.sum(L_w, axis=1, keepdims=True) # (S, 1)
            
            # Contribution from child w
            # M(s, x) = L_w(s, x) * exp_term + p_diff * sum_L_w(s)
            M = L_w * exp_term + p_diff * sum_L_w
            
            L_v = L_v * M
        
        L[v] = L_v
        
    # Total likelihood
    L_root = L[root]
    # Sum over root states
    total_prob = np.sum(L_root, axis=1)
    # Avoid log(0)
    total_prob = np.maximum(total_prob, 1e-300)
    log_L = np.sum(np.log(total_prob))
    
    return log_L


def _optimize_branch_lengths(children, root, edges, seqs, initial_lengths):
    """
    Optimize branch lengths using scipy.optimize.minimize_scalar.
    initial_lengths: dict node -> length
    """
    # List of internal branches (branches leading to internal nodes or root)
    # We optimize lengths of branches connecting to internal nodes.
    # Leaves have no length to optimize in this context (length is on the edge to parent).
    # So we optimize length for every node except the root.
    
    nodes_to_optimize = [n for n in children if n != root]
    
    # Map node index to parameter index
    param_idx = {n: i for i, n in enumerate(nodes_to_optimize)}
    x0 = np.array([max(initial_lengths[n], 1e-4) for n in nodes_to_optimize])
    
    def neg_log_likelihood(x):
        branch_lengths = {n: max(val, 1e-4) for n, val in zip(nodes_to_optimize, x)}
        # Root branch length is 0 or fixed
        branch_lengths[root] = 0.0
        try:
            ll = _calculate_likelihood(children, root, branch_lengths, seqs)
            return -ll
        except Exception:
            return 1e10
            
    # Use Nelder-Mead or Powell for speed? Nelder-Mead is robust.
    # Given low dimensionality (N-1), it should be fast.
    res = minimize_scalar(neg_log_likelihood, bounds=(1e-4, 5.0), method='bounded')
    if res.success:
        x_opt = res.x
        branch_lengths = {n: max(val, 1e-4) for n, val in zip(nodes_to_optimize, x_opt)}
        branch_lengths[root] = 0.0
        return branch_lengths, -res.fun
    else:
        return initial_lengths, 1e10


def _spr_search(children, root, edges, branch_lengths, seqs, max_time):
    """
    Perform SPR search to improve topology.
    """
    current_ll = _calculate_likelihood(children, root, branch_lengths, seqs)
    
    while _elapsed() < _DEADLINE - 10:
        improved = False
        # Generate all SPR moves
        # Prune a subtree at node u (u is not root)
        # Reattach to edge (v, w) where v is parent of w
        # We need to identify all edges and their directions
        
        # Rebuild adjacency and parent-child for current tree
        all_nodes = set(children.keys())
        adj = {node: [] for node in all_nodes}
        for u in children:
            for v in children[u]:
                adj[u].append(v)
                adj[v].append(u)
                
        # Find all edges (parent, child)
        parent_map = {root: None}
        def build_parent_map(node):
            for c in children[node]:
                parent_map[c] = node
                build_parent_map(c)
        build_parent_map(root)
        
        edges_list = [(parent_map[c], c) for c in children if c != root]
        
        for u in all_nodes:
            if u == root:
                continue
            # Prune subtree rooted at u
            # The edge to prune is (parent_map[u], u)
            p_u = parent_map[u]
            
            # Remove u from p_u's children
            children[p_u].remove(u)
            
            # Reattach u to every other edge (v, w)
            for v, w in edges_list:
                if v == u or w == u:
                    continue
                if v == p_u and w == u: # Original position
                    continue
                
                # Insert u between v and w
                # v -> u -> w
                # Remove w from v's children
                children[v].remove(w)
                # Add w to u's children
                children[u].append(w)
                # Add u to v's children
                children[v].append(u)
                
                # Update parent map for this move
                parent_map[u] = v
                parent_map[w] = u
                
                # Optimize branch lengths for the new topology
                # We can reuse old lengths as initial guess
                new_branch_lengths = {n: branch_lengths.get(n, 0.1) for n in all_nodes}
                new_branch_lengths[root] = 0.0
                
                # Quick optimization
                new_bl, new_ll = _optimize_branch_lengths(children, root, edges_list, seqs, new_branch_lengths)
                
                if new_ll > current_ll + 1e-6:
                    current_ll = new_ll
                    branch_lengths = new_bl
                    improved = True
                    # Restart search from new tree
                    break
                else:
                    # Revert
                    children[v].remove(u)
                    children[u].remove(w)
                    children[v].append(w)
                    parent_map[u] = None # Will be fixed
                    parent_map[w] = v
                    # Revert u's parent
                    children[p_u].append(u)
                    parent_map[u] = p_u
                    # Revert w's parent
                    # w was child of v, now it is.
                    
            if improved:
                break
            if _elapsed() > _DEADLINE - 10:
                break
                
        if not improved:
            break
            
    return children, root, branch_lengths, current_ll


def _tree_to_newick(children, root, labels):
    """Convert tree structure to Newick string."""
    def build_newick(node):
        if node < len(labels):
            return labels[node]
        kids = children[node]
        if not kids:
            return str(node) # Should not happen for internal
        parts = [build_newick(k) for k in kids]
        return "(" + ",".join(parts) + ")"
        
    return build_newick(root) + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    try:
        # 1. Distance Matrix
        dist = _jc_distance_matrix(sequences, labels)
        
        # 2. NJ Tree
        edges = _neighbor_joining(dist)
        
        # 3. Build Tree Structure
        children, root, edges = _build_tree_from_edges(edges, n)
        
        # 4. Initial Branch Lengths from Distances
        # Simple heuristic: average distance to leaves in the subtree
        branch_lengths = {node: 0.1 for node in children}
        branch_lengths[root] = 0.0
        
        # 5. Optimize Branch Lengths
        branch_lengths, ll = _optimize_branch_lengths(children, root, edges, _encode_sequences(sequences, labels), branch_lengths)
        
        # 6. SPR Search
        seqs = _encode_sequences(sequences, labels)
        children, root, branch_lengths, ll = _spr_search(children, root, edges, branch_lengths, seqs, _DEADLINE)
        
        # 7. Final Branch Length Optimization
        branch_lengths, ll = _optimize_branch_lengths(children, root, edges, seqs, branch_lengths)
        
        # 8. Convert to Newick
        newick = _tree_to_newick(children, root, labels)
        
        return newick
    except Exception as e:
        # Fallback to star tree
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
