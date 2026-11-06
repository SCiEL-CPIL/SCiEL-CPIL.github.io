"""Phylogenetic topology recovery: NJ initialization + ML-based NNI local search."""
import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255  # safety margin before 270s kill

def _check_deadline():
    return time.time() < _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _parse_newick(newick):
    """Parse a Newick string into a nested tuple structure."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    def parse(s):
        if s[0] == '(':
            depth = 0
            i = 1
            parts = []
            current = ''
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                    current += s[i]
                elif s[i] == ')':
                    depth -= 1
                    if depth == -1:
                        break
                    current += s[i]
                elif s[i] == ',' and depth == 0:
                    parts.append(current)
                    current = ''
                else:
                    current += s[i]
                i += 1
            if i < len(s):
                i += 1
            rest = s[i:]
            if rest.startswith(':'):
                j = 0
                while j < len(rest) and rest[j] not in (',', '(', ')', ';'):
                    j += 1
            children = [parse(p) for p in parts]
            return ('internal', children)
        else:
            label = s
            if ':' in label:
                label = label.split(':')[0]
            return ('leaf', label)
    
    return parse(newick)


def _build_adjacency(tree):
    """Build adjacency list from tree structure."""
    adj = {}
    node_types = {}
    counter = [0]
    
    def assign_id(node, parent_id=None):
        if node[0] == 'leaf':
            nid = ('leaf', node[1])
            node_types[nid] = ('leaf', node[1])
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            return nid
        else:
            nid = ('internal', counter[0])
            counter[0] += 1
            node_types[nid] = ('internal', None)
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            for child in node[1]:
                cid = assign_id(child, nid)
            return nid
    
    root = assign_id(tree)
    node_ids = list(adj.keys())
    return adj, node_ids, node_types


def _root_tree(adj, node_ids, node_types):
    """Root the tree and build parent/children structure."""
    root = None
    for nid in node_ids:
        if node_types[nid][0] == 'internal':
            root = nid
            break
    if root is None:
        root = node_ids[0]
    
    parent = {root: None}
    children = {nid: [] for nid in node_ids}
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                children[node].append(nbr)
                queue.append(nbr)
    
    return root, parent, children


def _postorder(root, children):
    """Get postorder traversal."""
    order = []
    def _post(node):
        for child in children[node]:
            _post(child)
        order.append(node)
    _post(root)
    return order


def _ml_score_tree_fast(adj, node_ids, node_types, seq_array, n_sites, base_length):
    """Compute ML log-likelihood using vectorized NumPy operations.
    
    seq_array: (n_taxa, n_sites) int array with values 0-3
    node_types: maps node_id -> ('leaf', label) or ('internal', None)
    """
    # Build rooted structure
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # Map leaf labels to indices in seq_array
    leaf_label_to_idx = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_label_to_idx[node_types[nid][1]] = len(leaf_label_to_idx)
    
    # Precompute transition probabilities
    exp_term = math.exp(-4.0 * base_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = (1.0 - exp_term) / 3.0
    
    prob = {}
    
    # Initialize leaves
    for nid in order:
        if node_types[nid][0] == 'leaf':
            label = node_types[nid][1]
            idx = leaf_label_to_idx[label]
            p = np.zeros((n_sites, 4), dtype=np.float64)
            obs = seq_array[idx]
            p[np.arange(n_sites), obs] = 1.0
            prob[nid] = p
        else:
            prob[nid] = None
    
    # Process internal nodes in postorder
    for nid in order:
        if node_types[nid][0] == 'leaf':
            continue
        
        child_list = children[nid]
        if not child_list:
            prob[nid] = np.ones((n_sites, 4), dtype=np.float64)
            continue
        
        result = np.ones((n_sites, 4), dtype=np.float64)
        
        for child in child_list:
            cp = prob[child]
            row_sum = cp.sum(axis=1, keepdims=True)
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp
            result *= child_contrib
        
        prob[nid] = result
    
    root_prob = prob[root].mean(axis=1)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll


def _ml_score_with_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, branch_lengths):
    """Compute ML log-likelihood with per-branch lengths.
    
    branch_lengths: dict mapping (node_a, node_b) -> length, where node_a < node_b
    """
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    leaf_label_to_idx = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_label_to_idx[node_types[nid][1]] = len(leaf_label_to_idx)
    
    prob = {}
    
    for nid in order:
        if node_types[nid][0] == 'leaf':
            label = node_types[nid][1]
            idx = leaf_label_to_idx[label]
            p = np.zeros((n_sites, 4), dtype=np.float64)
            obs = seq_array[idx]
            p[np.arange(n_sites), obs] = 1.0
            prob[nid] = p
        else:
            prob[nid] = None
    
    for nid in order:
        if node_types[nid][0] == 'leaf':
            continue
        
        child_list = children[nid]
        if not child_list:
            prob[nid] = np.ones((n_sites, 4), dtype=np.float64)
            continue
        
        result = np.ones((n_sites, 4), dtype=np.float64)
        
        for child in child_list:
            # Get branch length between nid and child
            key = tuple(sorted([nid, child]))
            bl = branch_lengths.get(key, 0.1)
            exp_term = math.exp(-4.0 * bl / 3.0)
            p_same = 0.25 + 0.75 * exp_term
            p_diff = (1.0 - exp_term) / 3.0
            
            cp = prob[child]
            row_sum = cp.sum(axis=1, keepdims=True)
            child_contrib = p_diff * row_sum + (p_same - p_diff) * cp
            result *= child_contrib
        
        prob[nid] = result
    
    root_prob = prob[root].mean(axis=1)
    root_prob_clipped = np.maximum(root_prob, 1e-300)
    total_ll = np.sum(np.log(root_prob_clipped))
    
    return total_ll


def _estimate_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels):
    """Estimate branch lengths using distance-based method.
    
    Returns dict mapping (node_a, node_b) -> length
    """
    # Get leaf set
    leaves = [nid for nid in node_ids if node_types[nid][0] == 'leaf']
    
    # Build edge list
    edges = []
    for u in node_ids:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    leaf_indices = {nid: i for i, nid in enumerate(leaves)}
    leaf_labels = {nid: node_types[nid][1] for nid in leaves}
    
    # For each edge, compute the set of leaves reachable from each endpoint
    # without going through the edge
    edge_leaf_sets = {}
    for u, v in edges:
        # Find leaves reachable from u without going through v
        visited_u = set()
        stack = [u]
        while stack:
            node = stack.pop()
            if node in visited_u:
                continue
            visited_u.add(node)
            for nbr in adj[node]:
                if nbr != v and nbr not in visited_u:
                    stack.append(nbr)
        
        # Find leaves reachable from v without going through u
        visited_v = set()
        stack = [v]
        while stack:
            node = stack.pop()
            if node in visited_v:
                continue
            visited_v.add(node)
            for nbr in adj[node]:
                if nbr != u and nbr not in visited_v:
                    stack.append(nbr)
        
        leaves_u = [nid for nid in visited_u if nid in leaf_indices]
        leaves_v = [nid for nid in visited_v if nid in leaf_indices]
        
        edge_leaf_sets[(u, v)] = (leaves_u, leaves_v)
    
    branch_lengths = {}
    # For each edge, estimate length as:
    # avg distance from leaves_u to leaves_v - (avg distance within leaves_u + avg distance within leaves_v) / 2
    # This is a rough estimate
    
    for u, v in edges:
        leaves_u, leaves_v = edge_leaf_sets[(u, v)]
        
        if not leaves_u or not leaves_v:
            branch_lengths[(u, v)] = 0.1
            continue
        
        # Average distance from leaves_u to leaves_v
        cross_dists = []
        for lu in leaves_u:
            for lv in leaves_v:
                label_u = leaf_labels[lu]
                label_v = leaf_labels[lv]
                d = distance_matrix.get((label_u, label_v), 0.0)
                cross_dists.append(d)
        
        avg_cross = sum(cross_dists) / len(cross_dists) if cross_dists else 0.1
        
        # Average distance within leaves_u
        if len(leaves_u) > 1:
            within_u = []
            for i, lu in enumerate(leaves_u):
                for lv in leaves_u[i+1:]:
                    label_u = leaf_labels[lu]
                    label_v = leaf_labels[lv]
                    d = distance_matrix.get((label_u, label_v), 0.0)
                    within_u.append(d)
            avg_within_u = sum(within_u) / len(within_u) if within_u else 0.0
        else:
            avg_within_u = 0.0
        
        # Average distance within leaves_v
        if len(leaves_v) > 1:
            within_v = []
            for i, lv in enumerate(leaves_v):
                for lv2 in leaves_v[i+1:]:
                    label_v = leaf_labels[lv]
                    label_v2 = leaf_labels[lv2]
                    d = distance_matrix.get((label_v, label_v2), 0.0)
                    within_v.append(d)
            avg_within_v = sum(within_v) / len(within_v) if within_v else 0.0
        else:
            avg_within_v = 0.0
        
        # Estimate branch length
        bl = max(0.01, (avg_cross - avg_within_u - avg_within_v) / 2.0)
        bl = min(bl, 1.0)  # Cap at 1.0
        branch_lengths[(u, v)] = bl
    
    return branch_lengths


def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure.
    
    For internal edge (u,v) where u has neighbors {v,a,b} and v has neighbors {u,c,d}:
    NNI1: swap b and c -> changes split from (a,b)|(c,d) to (a,c)|(b,d)
    NNI2: swap b and d -> changes split from (a,b)|(c,d) to (a,d)|(b,c)
    """
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        # Swap b and c
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        # Swap b and d
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj


def _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels, max_iterations=50):
    """Perform NNI local search to improve tree topology using ML scoring with per-branch lengths."""
    # Estimate initial branch lengths
    branch_lengths = _estimate_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels)
    current_ll = _ml_score_with_branch_lengths(adj, node_ids, node_types, seq_array, n_sites, branch_lengths)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        best_ll = current_ll
        best_adj = None
        best_bl = None
        
        # Get internal edges for NNI moves
        root, parent, children = _root_tree(adj, node_ids, node_types)
        
        # Find all internal edges
        internal_edges = []
        for u in node_ids:
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if u < v and node_types[v][0] == 'internal':
                    # Check that u and v are connected by an edge and both are internal
                    # For NNI, we need u to have neighbors {v, a, b} and v to have neighbors {u, c, d}
                    u_nbrs = [x for x in adj[u] if x != v]
                    v_nbrs = [x for x in adj[v] if x != u]
                    
                    if len(u_nbrs) == 2 and len(v_nbrs) == 2:
                        a, b = u_nbrs
                        c, d = v_nbrs
                        internal_edges.append((u, v, a, b, c, d))
        
        # Try all NNI moves
        for u, v, a, b, c, d in internal_edges:
            if not _check_deadline():
                break
            
            for move_type in [1, 2]:
                new_adj = _apply_nni(adj, node_types, u, v, a, b, c, d, move_type)
                
                # Quick check: ensure it's still a valid tree (2n-2 edges)
                edge_count = sum(len(s) for s in new_adj.values()) // 2
                if edge_count != len(node_ids) - 1:
                    continue
                
                new_bl = _estimate_branch_lengths(new_adj, node_ids, node_types, seq_array, n_sites, distance_matrix, labels)
                ll = _ml_score_with_branch_lengths(new_adj, node_ids, node_types, seq_array, n_sites, new_bl)
                
                if ll > best_ll + 1e-6:
                    best_ll = ll
                    best_adj = new_adj
                    best_bl = new_bl
        
        if best_adj is not None:
            adj = best_adj
            branch_lengths = best_bl
            current_ll = best_ll
            improved = True
    
    return adj


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    # Find a leaf to start from
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    # Build Newick by traversing from root
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in sorted(adj[nid], key=lambda x: str(x)):
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"


def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Step 1: Compute distance matrix using JC correction
    matrix = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                d = _jc_distance(sequences[a], sequences[b])
                matrix[(a, b)] = d
                matrix[(b, a)] = d
    
    # Step 2: Neighbor-Joining
    nj_nodes = list(labels)
    trees = {name: name for name in nj_nodes}
    
    while len(nj_nodes) > 3:
        if not _check_deadline():
            break
        n = len(nj_nodes)
        total = {a: sum(matrix[(a, b)] for b in nj_nodes if b != a) for a in nj_nodes}
        best = None
        for i, a in enumerate(nj_nodes):
            for b in nj_nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        if best is None:
            break
        
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nj_nodes)}_{a}_{b}"
        rest = [x for x in nj_nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nj_nodes = rest + [new]
        trees[new] = joined
    
    # Build initial Newick from the current tree structure
    # The root should be the last internal node created or the one with all leaves
    root_node = None
    import re
    for node in nj_nodes:
        subtree = trees[node]
        leaf_count = len(set(re.findall(r't\d+', subtree)))
        if leaf_count == n_taxa:
            root_node = node
            break
    
    if root_node is None:
        # Fallback: use the first node
        root_node = nj_nodes[0]
    
    newick = trees[root_node] + ";"
    
    # Ensure all taxa are present
    for label in labels:
        if label not in newick:
            # Fallback to simple NJ without pruning
            nodes = list(labels)
            trees = {name: name for name in labels}
            while len(nodes) > 3:
                if not _check_deadline():
                    break
                n = len(nodes)
                total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
                best = None
                for i, a in enumerate(nodes):
                    for b in nodes[i + 1:]:
                        criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                        if best is None or criterion < best[0]:
                            best = (criterion, a, b)
                if best is None:
                    break
                _, a, b = best
                joined = f"({trees[a]},{trees[b]})"
                new = f"n{len(nodes)}_{a}_{b}"
                rest = [x for x in nodes if x not in (a, b)]
                for x in rest:
                    matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
                nodes = rest + [new]
                trees[new] = joined
            newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Step 4: Prepare sequence data for vectorized ML scoring
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seq_array = np.zeros((n_taxa, n_sites), dtype=np.int32)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(n_sites):
            seq_array[i, j] = state_map[seq[j]]
    
    # Step 5: NNI local search with per-branch length ML scoring
    if _check_deadline():
        # Compute distance matrix for branch length estimation
        dist_matrix = {}
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i < j:
                    d = _jc_distance(sequences[a], sequences[b])
                    dist_matrix[(a, b)] = d
                    dist_matrix[(b, a)] = d
        
        # Use NNI search which is more efficient than SPR for topology recovery
        adj = _nni_local_search(adj, node_ids, node_types, seq_array, n_sites, dist_matrix, labels, max_iterations=50)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
