"""Quartet-based phylogenetic tree reconstruction with NNI refinement."""
import time
import math
import numpy as np
from itertools import combinations


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
    
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatches = diff.sum(axis=2)
    p = mismatches / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(dist, labels):
    """Standard neighbour joining with proper Newick construction."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 1:
        return labels[0]
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    dist_dict = {}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            d = float(dist[i, j])
            dist_dict[(nodes[i], nodes[j])] = d
            dist_dict[(nodes[j], nodes[i])] = d
    
    while len(nodes) > 3:
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
        len_a = 0.5 * d_ab + (row_sum[a] - row_sum[b]) / (2.0 * (n_cur - 2))
        len_b = d_ab - len_a
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        new_name = f"n{len(nodes)}_{a}_{b}"
        trees[new_name] = f"({trees[a]},{trees[b]})"
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            dist_dict[(new_name, x)] = d_new_x
            dist_dict[(x, new_name)] = d_new_x
        
        nodes = rest + [new_name]
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _parse_newick_adj(newick):
    """Parse Newick string into adjacency list structure.
    
    Returns:
        adj: dict mapping node_id -> list of neighbor node_ids
        node_to_label: dict mapping node_id -> taxon label (for leaves) or None (for internal)
        n_nodes: total number of nodes
        leaf_nodes: list of node_ids that are leaves
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    adj = {}
    node_to_label = {}
    next_id = [0]
    
    def next_node():
        nid = next_id[0]
        next_id[0] += 1
        adj[nid] = []
        node_to_label[nid] = None
        return nid
    
    pos = [0]
    
    def parse_subtree():
        # Skip whitespace
        while pos[0] < len(s) and s[pos[0]] == ' ':
            pos[0] += 1
        
        if pos[0] >= len(s):
            return None
        
        if s[pos[0]] == '(':
            # Internal node
            nid = next_node()
            pos[0] += 1  # skip '('
            
            # Parse children
            while True:
                child = parse_subtree()
                if child is not None:
                    adj[nid].append(child)
                    adj[child].append(nid)
                
                # Skip whitespace
                while pos[0] < len(s) and s[pos[0]] == ' ':
                    pos[0] += 1
                
                if pos[0] >= len(s):
                    break
                
                if s[pos[0]] == ',':
                    pos[0] += 1
                elif s[pos[0]] == ')':
                    pos[0] += 1
                    break
                else:
                    break
            
            # Skip branch length if present
            while pos[0] < len(s) and s[pos[0]] not in '(),;':
                pos[0] += 1
            
            return nid
        else:
            # Leaf node
            nid = next_node()
            # Parse label
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in '():,;':
                pos[0] += 1
            label = s[start:pos[0]]
            node_to_label[nid] = label
            
            # Skip branch length if present
            while pos[0] < len(s) and s[pos[0]] not in '(),;':
                pos[0] += 1
            
            return nid
    
    parse_subtree()
    
    n_nodes = next_id[0]
    leaf_nodes = [nid for nid in range(n_nodes) if node_to_label[nid] is not None]
    
    return adj, node_to_label, n_nodes, leaf_nodes


def _build_newick_from_adj(adj, node_to_label, n_nodes, leaf_nodes):
    """Build a Newick string from adjacency list."""
    # Find a root (pick a leaf)
    if not leaf_nodes:
        return ""
    
    root = leaf_nodes[0]
    
    # BFS/DFS to create parent-child relationships
    parent = {root: -1}
    visited = {root}
    stack = [root]
    
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in visited:
                parent[nb] = node
                visited.add(nb)
                stack.append(nb)
    
    # Build children lists
    children = {i: [] for i in range(n_nodes)}
    for node in range(n_nodes):
        if parent[node] != -1:
            children[parent[node]].append(node)
    
    def to_newick(node):
        kids = children[node]
        if not kids:
            return node_to_label[node]
        else:
            parts = [to_newick(k) for k in kids]
            return "(" + ",".join(parts) + ")"
    
    return to_newick(root) + ";"


def _get_subtree_taxa(adj, node, node_to_label, visited=None):
    """Get all taxon labels in the subtree rooted at node (away from parent)."""
    if visited is None:
        visited = set()
    
    if node in visited:
        return set()
    visited.add(node)
    
    if node_to_label[node] is not None:
        return {node_to_label[node]}
    
    result = set()
    for nb in adj[node]:
        if nb not in visited:
            result |= _get_subtree_taxa(adj, nb, node_to_label, visited)
    return result


def _nni_improve(newick, dist, labels, max_iterations=20, time_limit=None, start_time=None):
    """Apply NNI moves to improve the tree topology using distance-based scoring."""
    if time_limit is not None and start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return newick
    
    adj, node_to_label, n_nodes, leaf_nodes = _parse_newick_adj(newick)
    
    if n_nodes < 4:
        return newick
    
    # Map labels to indices in dist matrix
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Precompute taxon sets for each node (for efficient NNI scoring)
    def compute_taxon_sets():
        """Compute for each node the set of taxa in its subtree when rooted at a reference."""
        # Root at first leaf
        ref_root = leaf_nodes[0]
        taxon_sets = {}
        
        def compute_sets(node, parent):
            if node_to_label[node] is not None:
                taxon_sets[node] = {node_to_label[node]}
            else:
                s = set()
                for nb in adj[node]:
                    if nb != parent:
                        s |= compute_sets(nb, node)
                taxon_sets[node] = s
            return taxon_sets[node]
        
        compute_sets(ref_root, -1)
        return taxon_sets
    
    taxon_sets = compute_taxon_sets()
    
    def get_internal_edges():
        """Get all edges between two internal nodes."""
        edges = []
        seen = set()
        for u in range(n_nodes):
            if node_to_label[u] is not None:
                continue
            for v in adj[u]:
                if v <= u:
                    continue
                if node_to_label[v] is not None:
                    continue
                edges.append((u, v))
                seen.add((u, v))
                seen.add((v, u))
        return edges
    
    def nni_score(adj, taxon_sets):
        """Score a tree by sum of squared differences between tree and observed distances."""
        # For unrooted tree, we need to compute pairwise distances
        # Use the fact that for any edge, the tree distance between two leaves
        # is the sum of edges on the path between them.
        # For NNI, we can use a simpler metric: sum over all pairs of 
        # (observed_dist - tree_dist)^2
        # But computing tree_dist requires finding paths, which is expensive.
        
        # Alternative: use the "quartet" or "split" based scoring
        # For NNI, the change only affects splits involving the moved subtrees.
        
        # Simplest correct approach: compute all-pairs tree distances via BFS
        # But this is O(n^2 * n) which is fine for n<=32
        
        n_taxa = len(labels)
        leaf_list = list(leaf_nodes)
        leaf_to_label = {leaf_nodes[i]: labels[i] for i in range(n_taxa)}
        
        total_score = 0.0
        for i in range(n_taxa):
            for j in range(i+1, n_taxa):
                li = leaf_list[i]
                lj = leaf_list[j]
                label_i = leaf_to_label[li]
                label_j = leaf_to_label[lj]
                idx_i = label_to_idx[label_i]
                idx_j = label_to_idx[label_j]
                obs_d = dist[idx_i, idx_j]
                
                # BFS to find distance in tree (unit edges)
                # Actually, we don't have edge lengths in the adj structure
                # We need to use the observed distances directly
                
                # For NNI scoring, use a proxy: 
                # The NNI move changes which pairs are "close" in the tree.
                # Use the sum of distances across the cut.
                
                # Actually, let's use a different approach:
                # Score = sum over all pairs of (obs_dist) * indicator[they are on same side of some split]
                # This is getting complicated. Let's use the standard approach:
                # For each NNI candidate, compute the change in the sum of 
                # pairwise tree distances weighted by observed distances.
                
                # Simpler: use the "likelihood" proxy which for JC is 
                # related to the sum of pairwise distances.
                pass
        
        # Let's use a cleaner approach: 
        # The score for a tree T given distance matrix D is:
        # S(T) = sum_{i<j} D_ij * T_ij  (where T_ij is the tree distance)
        # Minimizing S is equivalent to maximizing correlation with D.
        # But we don't have T_ij without edge lengths.
        
        # For NNI, the key insight is that only the relative grouping changes.
        # We can use: for each pair of subtrees separated by an edge,
        # the "cost" is the sum of observed distances between taxa on opposite sides.
        
        # Let me just compute it properly with BFS for small n
        total = 0.0
        for i in range(n_taxa):
            li = leaf_list[i]
            # BFS from li
            from collections import deque
            dist_tree = {li: 0}
            q = deque([li])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if nb not in dist_tree:
                        dist_tree[nb] = dist_tree[node] + 1
                        q.append(nb)
            
            for j in range(i+1, n_taxa):
                lj = leaf_list[j]
                if lj in dist_tree:
                    t_d = dist_tree[lj]
                    label_i = leaf_to_label[li]
                    label_j = leaf_to_label[lj]
                    idx_i = label_to_idx[label_i]
                    idx_j = label_to_idx[label_j]
                    obs_d = dist[idx_i, idx_j]
                    total += (obs_d - t_d) ** 2
        return total
    
    best_score = nni_score(adj, taxon_sets)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time_limit is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        edges = get_internal_edges()
        
        for (u, v) in edges:
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) < 2 or len(v_neighbors) < 2:
                continue
            
            a, b = u_neighbors[0], u_neighbors[1]
            c, d = v_neighbors[0], v_neighbors[1]
            
            # Try NNI move 1: swap b and c
            # Original: u-a, u-b, u-v, v-c, v-d
            # Move 1:   u-a, u-c, u-v, v-b, v-d
            adj[u].remove(b)
            adj[u].append(c)
            adj[b].remove(u)
            adj[b].append(v)
            adj[v].remove(c)
            adj[v].append(b)
            adj[c].remove(v)
            adj[c].append(u)
            
            score1 = nni_score(adj, taxon_sets)
            
            if score1 < best_score - 1e-10:
                best_score = score1
                improved = True
            else:
                # Revert
                adj[u].remove(c)
                adj[u].append(b)
                adj[c].remove(u)
                adj[c].append(v)
                adj[v].remove(b)
                adj[v].append(c)
                adj[b].remove(v)
                adj[b].append(u)
            
            # Try NNI move 2: swap b and d
            # Move 2: u-a, u-d, u-v, v-b, v-c
            adj[u].remove(b)
            adj[u].append(d)
            adj[b].remove(u)
            adj[b].append(v)
            adj[v].remove(d)
            adj[v].append(b)
            adj[d].remove(v)
            adj[d].append(u)
            
            score2 = nni_score(adj, taxon_sets)
            
            if score2 < best_score - 1e-10:
                best_score = score2
                improved = True
            else:
                # Revert
                adj[u].remove(d)
                adj[u].append(b)
                adj[d].remove(u)
                adj[d].append(v)
                adj[v].remove(b)
                adj[v].append(d)
                adj[b].remove(v)
                adj[b].append(u)
    
    # Convert back to Newick
    result = _build_newick_from_adj(adj, node_to_label, n_nodes, leaf_nodes)
    return result


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    n_sites = len(sequences[labels[0]])
    
    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)
    
    if time.time() > deadline:
        return "(" + ",".join(labels) + ";"
    
    # Run NJ as initial tree
    result = _nj_tree(dist, labels)
    
    if time.time() > deadline - 5.0:
        return result
    
    # Apply NNI refinement
    try:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            result = _nni_improve(result, dist, labels, max_iterations=10,
                                  time_limit=remaining_time,
                                  start_time=start_time)
    except Exception:
        pass
    
    return result
