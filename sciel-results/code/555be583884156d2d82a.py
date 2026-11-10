"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _encode_sequences(sequences, labels):
    """Encode sequences to int8 array with A=0, C=1, G=2, T=3."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr
    return enc


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    enc = _encode_sequences(sequences, labels)
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
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


def _neighbor_joining(D, labels, deadline=None):
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
        if deadline is not None and time.time() > deadline:
            return "(" + ",".join(node_names) + ")"
        
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
        nodes: list of node IDs
        adj: adjacency list (dict node_id -> set of neighbor node_ids)
        leaf_to_node: dict mapping label -> node id
        node_to_label: dict mapping node id -> label (for leaves)
        root_id: the root node id
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
        """Parse subtree starting at position pos. Returns (node_id, new_pos)."""
        if pos >= len(s):
            raise ValueError("Unexpected end of Newick")
        
        if s[pos] == '(':
            # Internal node
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
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            
            return parent_id, pos
        else:
            # Leaf
            name_end = pos
            while name_end < len(s) and s[name_end] not in '():,':
                name_end += 1
            name = s[pos:name_end]
            leaf_id = next_node_id()
            leaf_to_node[name] = leaf_id
            node_to_label[leaf_id] = name
            pos = name_end
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            
            return leaf_id, pos
    
    root_id, _ = parse(newick, 0)
    
    return nodes, adj, leaf_to_node, node_to_label, root_id


def _get_splits_from_tree(adj, leaf_to_node, n_taxa):
    """
    Extract all non-trivial splits from an unrooted tree.
    Returns set of frozensets of frozensets.
    """
    all_taxa = set(leaf_to_node.keys())
    n = len(all_taxa)
    if n <= 2:
        return set()
    
    splits = set()
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    for u, v in edges:
        # BFS from u avoiding v to find all reachable leaves
        visited = set()
        queue = [u]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nb in adj[node]:
                if nb == v:
                    continue
                if nb not in visited:
                    queue.append(nb)
        
        side_a = set()
        for node in visited:
            if node in leaf_to_node:
                side_a.add(leaf_to_node[node])
        
        side_b = all_taxa - side_a
        
        if len(side_a) == 0 or len(side_b) == 0:
            continue
        if len(side_a) == n or len(side_b) == n:
            continue
        
        a = frozenset(sorted(side_a))
        b = frozenset(sorted(side_b))
        if a < b:
            split = frozenset((a, b))
        else:
            split = frozenset((b, a))
        splits.add(split)
    
    return splits


def _compute_jc_likelihood(adj, leaf_to_node, enc, n_taxa, n_sites, branch_lengths=None):
    """
    Compute log-likelihood using Felsenstein's pruning with JC model.
    Returns log-likelihood.
    """
    all_nodes = set(adj.keys())
    leaf_nodes = set(leaf_to_node.values())
    internal_nodes = all_nodes - leaf_nodes
    
    # Root at an internal node
    if not internal_nodes:
        root = list(all_nodes)[0]
    else:
        root = list(internal_nodes)[0]
    
    # Build parent-child
    parent = {root: None}
    children = {root: []}
    visited = {root}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v not in children:
                    children[v] = []
                children[u].append(v)
                queue.append(v)
    
    # Post-order
    post_order = []
    def dfs(u):
        for c in children.get(u, []):
            dfs(c)
        post_order.append(u)
    
    dfs(root)
    
    # Get all edges
    edges = []
    for u in all_nodes:
        for v in adj.get(u, set()):
            if u < v:
                edges.append((u, v))
    
    if branch_lengths is None:
        branch_lengths = {e: 0.1 for e in edges}
    
    # Map leaf nodes to taxon indices
    leaf_to_taxon = {}
    for label, node_id in leaf_to_node.items():
        taxon_idx = int(label[1:])
        leaf_to_taxon[node_id] = taxon_idx
    
    # Felsenstein's algorithm, vectorized over sites
    L = {}
    
    for u in post_order:
        if u in leaf_nodes:
            taxon_idx = leaf_to_taxon[u]
            L[u] = np.zeros((4, n_sites))
            for a in range(4):
                L[u][a] = (enc[taxon_idx] == a).astype(np.float64)
        else:
            L[u] = np.ones((4, n_sites))
            for c in children.get(u, []):
                if (u, c) in branch_lengths:
                    bl = branch_lengths[(u, c)]
                elif (c, u) in branch_lengths:
                    bl = branch_lengths[(c, u)]
                else:
                    bl = 0.1
                
                exp_term = np.exp(-4.0 * bl / 3.0)
                P_ii = 0.25 + 0.75 * exp_term
                P_ij = 0.25 - 0.25 * exp_term
                
                sum_Lc = np.sum(L[c], axis=0)
                for a in range(4):
                    L[u][a] += P_ii * L[c][a] + P_ij * (sum_Lc - L[c][a])
            
            max_val = np.max(L[u])
            if max_val > 1e-300:
                L[u] /= max_val
    
    root_ll = np.sum(L[root], axis=0)
    log_likelihood = np.sum(np.log(np.maximum(root_ll, 1e-300)))
    
    return log_likelihood


def _optimize_branch_lengths(adj, leaf_to_node, enc, n_taxa, n_sites, max_iter=5, tol=1e-3):
    """
    Optimize branch lengths for a given topology using coordinate descent.
    Returns optimized branch lengths and log-likelihood.
    """
    all_nodes = set(adj.keys())
    leaf_nodes = set(leaf_to_node.values())
    
    edges = []
    for u in all_nodes:
        for v in adj.get(u, set()):
            if u < v:
                edges.append((u, v))
    
    if not edges:
        return {}, 0.0
    
    branch_lengths = {e: 0.1 for e in edges}
    
    best_ll = _compute_jc_likelihood(adj, leaf_to_node, enc, n_taxa, n_sites, branch_lengths)
    
    for iteration in range(max_iter):
        improved = False
        for edge in edges:
            current_bl = branch_lengths[edge]
            best_bl = current_bl
            best_edge_ll = -np.inf
            
            candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            for factor in [0.5, 0.7, 1.3, 2.0]:
                candidates.append(current_bl * factor)
            
            for bl in candidates:
                if bl < 0 or bl > 10:
                    continue
                branch_lengths[edge] = bl
                ll = _compute_jc_likelihood(adj, leaf_to_node, enc, n_taxa, n_sites, branch_lengths)
                if ll > best_edge_ll:
                    best_edge_ll = ll
                    best_bl = bl
            
            branch_lengths[edge] = best_bl
            if abs(best_bl - current_bl) > tol:
                improved = True
        
        current_ll = _compute_jc_likelihood(adj, leaf_to_node, enc, n_taxa, n_sites, branch_lengths)
        if current_ll > best_ll:
            best_ll = current_ll
        if not improved:
            break
    
    return branch_lengths, best_ll


def _tree_to_newick(adj, leaf_to_node, node_to_label, root_id):
    """Convert tree adjacency back to Newick string."""
    if not adj:
        return ""
    
    parent = {root_id: None}
    children = {root_id: []}
    visited = {root_id}
    queue = [root_id]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v not in children:
                    children[v] = []
                children[u].append(v)
                queue.append(v)
    
    def build_newick(u):
        if u in leaf_to_node:
            return leaf_to_node[u]
        if not children.get(u, []):
            return f"({u})"
        child_strs = [build_newick(c) for c in children[u]]
        return f"({','.join(child_strs)})"
    
    return build_newick(root_id)


def _spr_improve(newick, sequences, labels, deadline):
    """
    Improve a Newick tree using SPR local search with likelihood scoring.
    Returns improved Newick string.
    """
    start_time = time.time()
    remaining_time = deadline - start_time
    
    if remaining_time < 30:
        return newick
    
    nodes, adj, leaf_to_node, node_to_label, root_id = _parse_newick_to_tree(newick, labels)
    n_nodes = len(nodes)
    n_taxa = len(labels)
    
    seq_len = len(sequences[labels[0]])
    enc = _encode_sequences(sequences, labels)
    
    # Optimize branch lengths for initial tree
    bl, current_ll = _optimize_branch_lengths(adj, leaf_to_node, enc, n_taxa, seq_len, max_iter=5)
    
    best_newick = newick
    best_ll = current_ll
    best_bl = bl
    
    # Build parent-child from root
    parent = {root_id: None}
    children = {root_id: []}
    visited = {root_id}
    queue = [root_id]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, set()):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v not in children:
                    children[v] = []
                children[u].append(v)
                queue.append(v)
    
    leaf_nodes = set(leaf_to_node.values())
    
    # Get all edges
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    max_moves = 100
    moves_done = 0
    
    for iteration in range(5):
        if time.time() - start_time > remaining_time - 10:
            break
        if moves_done >= max_moves:
            break
        
        improved = False
        for edge_idx, (u, v) in enumerate(edges):
            if time.time() - start_time > remaining_time - 10 or moves_done >= max_moves:
                break
            
            if parent.get(v) != u:
                continue
            
            # Get the subtree rooted at v
            subtree_nodes = set()
            stack = [v]
            while stack:
                node = stack.pop()
                if node in subtree_nodes:
                    continue
                subtree_nodes.add(node)
                for c in children.get(node, []):
                    stack.append(c)
            
            # Try regrafting at other edges
            for reg_edge_idx, (ru, rv) in enumerate(edges):
                if reg_edge_idx == edge_idx:
                    continue
                if time.time() - start_time > remaining_time - 10 or moves_done >= max_moves:
                    break
                
                if ru in subtree_nodes or rv in subtree_nodes:
                    continue
                
                new_adj = {k: set(v2) for k, v2 in adj.items()}
                
                new_adj[u].discard(v)
                new_adj[v].discard(u)
                
                if ru in new_adj[v]:
                    continue
                
                new_adj[v].add(ru)
                new_adj[ru].add(v)
                
                n_edges = sum(len(s) for s in new_adj.values()) // 2
                if n_edges != n_nodes - 1:
                    continue
                
                visited_set = {u}
                q = [u]
                while q:
                    node = q.pop(0)
                    for nb in new_adj[node]:
                        if nb not in visited_set:
                            visited_set.add(nb)
                            q.append(nb)
                
                if len(visited_set) != n_nodes:
                    continue
                
                moves_done += 1
                
                # Compute likelihood for new topology
                new_bl, new_ll = _optimize_branch_lengths(new_adj, leaf_to_node, enc, n_taxa, seq_len, max_iter=3)
                
                if new_ll > best_ll:
                    adj = new_adj
                    best_ll = new_ll
                    best_bl = new_bl
                    best_newick = _tree_to_newick(adj, leaf_to_node, node_to_label, root_id)
                    improved = True
                    
                    # Rebuild parent-child and edges
                    parent = {root_id: None}
                    children = {root_id: []}
                    visited = {root_id}
                    queue = [root_id]
                    while queue:
                        uu = queue.pop(0)
                        for vv in adj.get(uu, set()):
                            if vv not in visited:
                                visited.add(vv)
                                parent[vv] = uu
                                if vv not in children:
                                    children[vv] = []
                                children[uu].append(vv)
                                queue.append(vv)
                    
                    edges = []
                    for uu in adj:
                        for vv in adj[uu]:
                            if uu < vv:
                                edges.append((uu, vv))
                    break
        
        if not improved:
            break
    
    return best_newick


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
    elapsed = time.time() - start_time
    
    # Run neighbor joining
    result = _neighbor_joining(D, labels, deadline=start_time + deadline - 5)
    
    # Try to improve with SPR if time allows
    remaining = deadline - time.time()
    if remaining > 30 and n <= 30:
        improved = _spr_improve(result, sequences, labels, deadline)
        if improved is not None:
            result = improved
    
    return result + ";"
# EVOLVE-BLOCK-END
