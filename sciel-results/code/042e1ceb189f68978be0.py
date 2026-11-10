"""Neighbour-joining with proper NNI refinement using quartet scoring, vectorized JC distances."""
import time
import numpy as np


def _parse_newick(newick):
    """Parse a Newick string into an adjacency list (undirected tree).
    Returns (adj, leaves, labels_map) where:
      adj: dict node_id -> set of neighbor node_ids
      leaves: set of leaf node_ids (taxon names)
      labels_map: dict taxon_name -> node_id (identity for leaves)
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    adj = {}
    leaves = set()
    counter = [0]
    
    def next_id():
        nid = f"_n{counter[0]}"
        counter[0] += 1
        adj[nid] = set()
        return nid
    
    pos = [0]
    
    def parse():
        while pos[0] < len(s) and s[pos[0]] in ' \t\n':
            pos[0] += 1
        if pos[0] >= len(s):
            return None
        if s[pos[0]] == '(':
            pos[0] += 1
            node = next_id()
            children = []
            while True:
                child = parse()
                if child is None:
                    break
                children.append(child)
                while pos[0] < len(s) and s[pos[0]] in ' \t\n':
                    pos[0] += 1
                if pos[0] < len(s) and s[pos[0]] == ',':
                    pos[0] += 1
                elif pos[0] < len(s) and s[pos[0]] == ')':
                    pos[0] += 1
                    break
                else:
                    break
            # Skip branch length / label after ')'
            while pos[0] < len(s) and s[pos[0]] not in '(),;':
                pos[0] += 1
            for c in children:
                adj[node].add(c)
                adj[c].add(node)
            return node
        else:
            # Leaf
            name = ''
            while pos[0] < len(s) and s[pos[0]] not in '():,;':
                name += s[pos[0]]
                pos[0] += 1
            # Skip branch length
            while pos[0] < len(s) and s[pos[0]] not in '():,;':
                pos[0] += 1
            adj[name] = set()
            leaves.add(name)
            return name
    
    root = parse()
    return adj, leaves


def _to_newick(adj, leaves):
    """Convert adjacency list to Newick string (unrooted, arbitrary root)."""
    # Find a leaf to root from
    if not leaves:
        return ""
    root = next(iter(leaves))
    
    # Build parent/children via BFS from root
    parent = {root: None}
    children = {n: [] for n in adj}
    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                children[node].append(nb)
                stack.append(nb)
    
    def build(nid):
        kids = children[nid]
        if not kids:
            return nid  # leaf
        parts = [build(c) for c in kids]
        return "(" + ",".join(parts) + ")"
    
    return build(root) + ";"


def _get_internal_edges(adj, leaves):
    """Get all edges between two internal nodes."""
    edges = []
    seen = set()
    for u in adj:
        if u in leaves:
            continue
        for v in adj[u]:
            if v in leaves:
                continue
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append((u, v))
    return edges


def _subtree_taxa(node, adj, leaves, parent=None):
    """Get all leaf taxa in the component containing 'node' when edge (node, parent) is removed."""
    if parent is None:
        # Get all leaves
        return set(leaves)
    # BFS from node, not going through parent
    visited = {node}
    stack = [node]
    taxa = set()
    while stack:
        cur = stack.pop()
        if cur in leaves:
            taxa.add(cur)
        for nb in adj[cur]:
            if nb != parent and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return taxa


def _nni_improve(adj, leaves, dist_matrix, labels, max_iterations=20, time_limit=10.0, start_time=None):
    """NNI local search using quartet score based on distance matrix.
    
    For each internal edge (u,v), the four subtrees are the components attached to u and v
    (excluding the edge u-v). We compute the quartet score for the three possible topologies
    and keep the best one.
    
    Score for topology (A,B)|(C,D) is: d(A,C)+d(A,D)+d(B,C)+d(B,D) - d(A,B)-d(C,D)
    where d(X,Y) is the average distance between taxa in subtree X and subtree Y.
    Actually, for NNI we want to minimize the total tree length, which is approximated by
    the sum of distances. The standard NNI criterion uses:
      current: (A,B) on one side, (C,D) on other -> score = d(A,C)+d(A,D)+d(B,C)+d(B,D)
      alt1: (A,C) and (B,D) -> score = d(A,B)+d(A,D)+d(C,B)+d(C,D)  
      alt2: (A,D) and (B,C) -> score = d(A,B)+d(A,C)+d(D,B)+d(D,C)
    We pick the minimum.
    """
    if time_limit is not None and start_time is not None:
        if time.time() - start_time > time_limit:
            return adj
    
    # Precompute label index map
    label_idx = {lab: i for i, lab in enumerate(labels)}
    
    def avg_dist_between_sets(set1, set2):
        """Average distance between all pairs of taxa in set1 and set2."""
        if not set1 or not set2:
            return 0.0
        total = 0.0
        count = 0
        for t1 in set1:
            i1 = label_idx[t1]
            for t2 in set2:
                i2 = label_idx[t2]
                total += dist_matrix[i1, i2]
                count += 1
        return total / count if count > 0 else 0.0
    
    def sum_dist_between_sets(set1, set2):
        """Sum of distances between all pairs of taxa in set1 and set2."""
        if not set1 or not set2:
            return 0.0
        total = 0.0
        for t1 in set1:
            i1 = label_idx[t1]
            for t2 in set2:
                i2 = label_idx[t2]
                total += dist_matrix[i1, i2]
        return total
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        edges = _get_internal_edges(adj, leaves)
        
        for (u, v) in edges:
            # Get the four subtrees
            u_nbs = [x for x in adj[u] if x != v]
            v_nbs = [x for x in adj[v] if x != u]
            
            if len(u_nbs) != 2 or len(v_nbs) != 2:
                continue
            
            a, b = u_nbs
            c, d = v_nbs
            
            # Get taxa sets for each subtree
            set_a = _subtree_taxa(a, adj, leaves, u)
            set_b = _subtree_taxa(b, adj, leaves, u)
            set_c = _subtree_taxa(c, adj, leaves, v)
            set_d = _subtree_taxa(d, adj, leaves, v)
            
            # Compute quartet scores
            # Current: (A,B)|(C,D)
            score_current = (sum_dist_between_sets(set_a, set_c) + 
                            sum_dist_between_sets(set_a, set_d) +
                            sum_dist_between_sets(set_b, set_c) +
                            sum_dist_between_sets(set_b, set_d))
            
            # Alt 1: (A,C)|(B,D)
            score_alt1 = (sum_dist_between_sets(set_a, set_b) +
                         sum_dist_between_sets(set_a, set_d) +
                         sum_dist_between_sets(set_c, set_b) +
                         sum_dist_between_sets(set_c, set_d))
            
            # Alt 2: (A,D)|(B,C)
            score_alt2 = (sum_dist_between_sets(set_a, set_b) +
                         sum_dist_between_sets(set_a, set_c) +
                         sum_dist_between_sets(set_d, set_b) +
                         sum_dist_between_sets(set_d, set_c))
            
            min_score = min(score_current, score_alt1, score_alt2)
            
            if min_score < score_current - 1e-10:
                if min_score == score_alt1:
                    # Swap b and c: u connects to a,c; v connects to b,d
                    adj[u].discard(b)
                    adj[b].discard(u)
                    adj[v].discard(c)
                    adj[c].discard(v)
                    
                    adj[u].add(c)
                    adj[c].add(u)
                    adj[v].add(b)
                    adj[b].add(v)
                elif min_score == score_alt2:
                    # Swap b and d: u connects to a,d; v connects to b,c
                    adj[u].discard(b)
                    adj[b].discard(u)
                    adj[v].discard(d)
                    adj[d].discard(v)
                    
                    adj[u].add(d)
                    adj[d].add(u)
                    adj[v].add(b)
                    adj[b].add(v)
                
                improved = True
    
    return adj


def _jc_dist_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using numpy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    
    # Pairwise mismatches using broadcasting
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatches = diff.sum(axis=2)
    p = mismatches / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(dist, labels):
    """Standard neighbour joining returning adjacency list."""
    n = len(labels)
    if n == 1:
        return {labels[0]: set()}, {labels[0]}
    if n == 2:
        return {labels[0]: {labels[1]}, labels[1]: {labels[0]}}, {labels[0], labels[1]}
    
    # Build initial distance dict
    dist_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            dist_dict[(labels[i], labels[j])] = d
            dist_dict[(labels[j], labels[i])] = d
    
    nodes = list(labels)
    adj = {name: set() for name in labels}
    trees = {name: name for name in labels}
    
    while len(nodes) > 2:
        n_cur = len(nodes)
        row_sum = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if a != b:
                    s += dist_dict[(a, b)]
            row_sum[a] = s
        
        best_criterion = float('inf')
        best_a = best_b = None
        for i_idx in range(len(nodes)):
            a = nodes[i_idx]
            for j_idx in range(i_idx + 1, len(nodes)):
                b = nodes[j_idx]
                criterion = (n_cur - 2) * dist_dict[(a, b)] - row_sum[a] - row_sum[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
        
        a, b = best_a, best_b
        d_ab = dist_dict[(a, b)]
        
        # Create new internal node
        new_name = f"_nj{len(nodes)}"
        adj[new_name] = {a, b}
        adj[a].add(new_name)
        adj[b].add(new_name)
        
        # Update distances
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            dist_dict[(new_name, x)] = d_new_x
            dist_dict[(x, new_name)] = d_new_x
        
        nodes = rest + [new_name]
    
    # Final: two nodes connected
    if len(nodes) == 2:
        adj[nodes[0]].add(nodes[1])
        adj[nodes[1]].add(nodes[0])
    
    leaves = set(labels)
    return adj, leaves


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 20 instances in 270s wall-clock
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > deadline:
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    adj, leaves = _nj_tree(dist, labels)
    
    # Check deadline before NNI
    remaining = deadline - time.time() - 2.0
    if remaining <= 0:
        return _to_newick(adj, leaves)
    
    # Apply NNI refinement
    try:
        # Use a time limit that scales with problem size but stays within budget
        nni_time_limit = min(remaining, 5.0)
        max_iter = 20 if n <= 20 else 10
        adj = _nni_improve(adj, leaves, dist, labels, 
                          max_iterations=max_iter,
                          time_limit=nni_time_limit,
                          start_time=start_time)
    except Exception:
        pass
    
    return _to_newick(adj, leaves)
