"""Rewritten program for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 265.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _build_jc_dist_matrix(sequences, labels):
    """Build pairwise JC distance matrix."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    seqs = np.zeros((n, L), dtype=np.int8)
    for i in range(n):
        s = sequences[labels[i]]
        for j in range(L):
            seqs[i, j] = seq_map[s[j]]
    
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(seqs[i] != seqs[j])
            p = diffs / L
            if p >= 0.75:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build_tree(dist, labels):
    """
    Correct Neighbor Joining. Returns (root, children, edges, leaf_set).
    """
    n = len(dist)
    if n <= 2:
        return None
    
    active = list(range(n))
    d = dist.copy()
    next_node = n
    edges = []
    
    while len(active) > 3:
        m = len(active)
        
        r = np.zeros(m)
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(m):
                if idx_i != idx_j:
                    j = active[idx_j]
                    r[idx_i] += d[i, j]
        
        best_criterion = float('inf')
        best_pair = None
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                criterion = (m - 2) * d[i, j] - r[idx_i] - r[idx_j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        new_node = next_node
        next_node += 1
        
        edges.append((i, new_node))
        edges.append((j, new_node))
        
        for k in active:
            if k == i or k == j:
                continue
            d[new_node, k] = d[k, new_node] = 0.5 * (d[i, k] + d[j, k] - d[i, j])
        
        active = [k for k in active if k != i and k != j]
        active.append(new_node)
        
        if _elapsed() > _DEADLINE - 5:
            break
    
    if len(active) == 3:
        a, b, c = active
        center = next_node
        next_node += 1
        edges.append((a, center))
        edges.append((b, center))
        edges.append((c, center))
    elif len(active) == 2:
        a, b = active
        edges.append((a, b))
    
    return edges


def _build_newick_from_edges(edges, labels):
    """Build Newick from edge list."""
    n = len(labels)
    if not edges:
        return "(" + ",".join(labels) + ");"
    
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
    
    leaf_set = set(range(n))
    start = None
    for node in adj:
        if node in leaf_set:
            start = node
            break
    if start is None:
        start = list(adj.keys())[0] if adj else 0
    
    node_name = {i: labels[i] for i in range(n)}
    
    def _build_newick(node, parent):
        if node in leaf_set:
            return node_name[node]
        neighbors = [nb for nb in adj.get(node, []) if nb != parent]
        if not neighbors:
            return node_name.get(node, f"n{node}")
        if len(neighbors) == 1:
            return "(" + _build_newick(neighbors[0], node) + ")"
        parts = [_build_newick(nb, node) for nb in neighbors]
        return "(" + ",".join(parts) + ")"
    
    newick = _build_newick(start, -1)
    return newick + ";"


def _compute_tree_length_from_edges(edges, dist, n):
    """
    Compute the total branch length of a tree given edges and distance matrix.
    For each edge (u,v), the branch length is estimated as:
    0.5 * (sum_{i in S, j not in S} d(i,j) - sum_{i in S, j in S} d(i,j) - sum_{i not in S, j not in S} d(i,j)) / (|S| * (n-|S|))
    where S is the set of leaves on one side of the edge.
    """
    # Build adjacency
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
    
    leaf_set = set(range(n))
    
    # For each edge, compute the split and the branch length
    total_length = 0.0
    for u, v in edges:
        # Get leaves on u's side (excluding v)
        def _get_leaves(node, parent):
            if node in leaf_set:
                return {node}
            leaves = set()
            for nb in adj.get(node, []):
                if nb != parent:
                    leaves |= _get_leaves(nb, node)
            return leaves
        
        side_u = _get_leaves(u, v)
        side_v = _get_leaves(v, u)
        
        # Use the smaller side for efficiency
        if len(side_u) > len(side_v):
            side_u, side_v = side_v, side_u
        
        if not side_u or len(side_u) == n:
            continue
        
        S = side_u
        S_comp = set(range(n)) - S
        
        # Compute sums
        sum_cross = 0.0
        sum_S = 0.0
        sum_S_comp = 0.0
        
        for i in S:
            for j in S:
                if i != j:
                    sum_S += dist[i, j]
            for j in S_comp:
                sum_cross += dist[i, j]
        
        for i in S_comp:
            for j in S_comp:
                if i != j:
                    sum_S_comp += dist[i, j]
        
        a = len(S)
        b = n - a
        
        if a == 0 or b == 0:
            continue
        
        # Branch length formula
        # L = 0.5 * (sum_cross - sum_S/(2*a) - sum_S_comp/(2*b)) / (a*b)
        # Actually, the standard formula is:
        # branch_length = (1/(2*a*b)) * (sum_{i in S, j in S_comp} d(i,j) - (1/(2*a)) * sum_{i,j in S} d(i,j) - (1/(2*b)) * sum_{i,j in S_comp} d(i,j))
        
        bl = (1.0 / (2.0 * a * b)) * (sum_cross - sum_S / (2.0 * a) - sum_S_comp / (2.0 * b))
        if bl > 0:
            total_length += bl
    
    return total_length


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]});"
    
    try:
        dist = _build_jc_dist_matrix(sequences, labels)
        edges = _nj_build_tree(dist, labels)
        
        if edges is None or not edges:
            return "(" + ",".join(labels) + ");"
        
        newick = _build_newick_from_edges(edges, labels)
        
        for label in labels:
            if label not in newick:
                return "(" + ",".join(labels) + ");"
        
        return newick
        
    except Exception:
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
