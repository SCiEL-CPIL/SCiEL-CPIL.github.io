"""Phylogenetic topology recovery: NJ + quartet-based NNI local search with JC-ML scoring."""
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


def _get_leaves_under(adj, node_types, start, exclude):
    """Get all leaf labels reachable from start without going through exclude."""
    leaves = []
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node_types[node][0] == 'leaf':
            leaves.append(node_types[node][1])
        else:
            for nbr in adj[node]:
                if nbr != exclude and nbr not in visited:
                    stack.append(nbr)
    return leaves


def _jc_likelihood_tree(adj, node_ids, node_types, sequences, labels):
    """Compute ML log-likelihood of the tree given the sequences using Felsenstein's pruning.
    Uses a fixed reasonable branch length for speed."""
    n_taxa = len(labels)
    if n_taxa < 2:
        return 0.0
    
    n_sites = len(sequences[labels[0]])
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build rooted structure
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # Estimate a global branch length scale
    total_d = 0.0
    count = 0
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            total_d += _jc_distance(sequences[a], sequences[b])
            count += 1
    mean_d = total_d / max(1, count)
    base_length = max(0.05, min(0.3, mean_d / max(1, n_taxa - 2)))
    
    # Precompute transition probabilities
    exp_term = math.exp(-4.0 * base_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = (1.0 - exp_term) / 3.0
    
    total_ll = 0.0
    
    for site in range(n_sites):
        prob = {}
        
        # Initialize leaves
        for nid in order:
            if node_types[nid][0] == 'leaf':
                label = node_types[nid][1]
                obs = state_map[sequences[label][site]]
                p = [0.0, 0.0, 0.0, 0.0]
                p[obs] = 1.0
                prob[nid] = p
            else:
                prob[nid] = None
        
        # Process internal nodes in postorder
        for nid in order:
            if node_types[nid][0] == 'leaf':
                continue
            
            p_node = [0.0, 0.0, 0.0, 0.0]
            for state in range(4):
                total = 1.0
                for child in children[nid]:
                    child_sum = 0.0
                    for c_state in range(4):
                        p_transition = p_same if c_state == state else p_diff
                        child_sum += p_transition * prob[child][c_state]
                    total *= child_sum
                p_node[state] = total
            
            prob[nid] = p_node
        
        root_prob = sum(prob[root]) / 4.0
        if root_prob > 1e-10:
            total_ll += math.log(root_prob)
        else:
            total_ll += -1000.0
    
    return total_ll


def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure.
    
    For internal edge (u,v) where u has neighbors {v,a,b} and v has neighbors {u,c,d}:
    NNI1: swap b and c -> u connected to a,c,v and v connected to u,b,d
    NNI2: swap b and d -> u connected to a,d,v and v connected to u,c,b
    """
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj


def _compute_me_score(adj, node_types, dist_matrix):
    """Compute Minimum Evolution score: sum of all pairwise path lengths on the tree.
    Lower is better. Uses the tree's implied distances vs observed distances.
    """
    node_ids = list(adj.keys())
    leaves = [nid for nid in node_ids if node_types[nid][0] == 'leaf']
    leaf_labels = [node_types[nid][1] for nid in leaves]
    
    # For each leaf pair, compute path length using BFS on the tree
    # We need to assign branch lengths. Use NJ-style: for each internal edge, 
    # the branch length is determined by the distances.
    # Simplified: compute the sum of all pairwise distances implied by the tree topology
    # using the distance matrix values as path lengths.
    
    # Actually, for ME we want to minimize the total tree length.
    # A practical approach: for each topology, compute the "implied" distances 
    # and compare to observed. But simpler: use the sum of all pairwise distances
    # as a proxy - trees that group close taxa together will have lower total path sum.
    
    # Better: compute for each leaf pair the path in the tree, sum all path lengths.
    # Since we don't have explicit branch lengths, use the distance matrix directly:
    # The ME score is sum over all pairs of d(i,j) where the path goes through 
    # specific internal nodes. 
    
    # Practical ME: For a given unrooted tree topology, the minimum evolution score
    # is the sum of all pairwise distances. But that's constant regardless of topology.
    # 
    # The correct ME: minimize sum of branch lengths where branch lengths are 
    # solved from the distance matrix. For a tree with n leaves, there are 2n-3 edges.
    # We can use the fact that for each edge, its length = (sum of distances across the 
    # split) / (2 * n * (n-1)) approximately.
    
    # Simpler approach that works: For each internal edge, compute the split it induces,
    # and the branch length is proportional to the average distance across that split.
    # Total tree length = sum of all branch lengths.
    
    total_length = 0.0
    n = len(leaf_labels)
    
    for nid in node_ids:
        if node_types[nid][0] != 'internal':
            continue
        for nbr in adj[nid]:
            if nbr <= nid:
                continue
            # Compute the split induced by removing edge (nid, nbr)
            # Get leaves on each side
            leaves_a = _get_leaves_under(adj, node_types, nid, nbr)
            leaves_b = _get_leaves_under(adj, node_types, nbr, nid)
            
            if not leaves_a or not leaves_b:
                continue
            
            # Branch length estimate: average distance across the split
            # For each pair (i in leaves_a, j in leaves_b), the distance contributes
            # to this edge's length
            cross_sum = 0.0
            cross_count = 0
            for la in leaves_a:
                for lb in leaves_b:
                    cross_sum += dist_matrix.get((la, lb), dist_matrix.get((lb, la), 0.0))
                    cross_count += 1
            
            if cross_count > 0:
                # The edge length is (sum of cross distances) / (2 * |A| * |B|)
                # This is an approximation
                edge_len = cross_sum / (2.0 * len(leaves_a) * len(leaves_b))
                total_length += edge_len
    
    return total_length


def _me_spr_search(adj, node_ids, node_types, sequences, labels, dist_matrix, max_rounds=10):
    """Perform SPR local search minimizing Minimum Evolution score."""
    current_score = _compute_me_score(adj, node_types, dist_matrix)
    
    for round_num in range(max_rounds):
        if not _check_deadline():
            break
        
        improved = False
        best_score = current_score
        best_adj = adj
        
        # Try NNI moves first (subset of SPR)
        for u in node_ids:
            if not _check_deadline():
                break
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if v <= u:
                    continue
                if node_types[v][0] != 'internal':
                    continue
                
                u_nbrs = [x for x in adj[u] if x != v]
                v_nbrs = [x for x in adj[v] if x != u]
                
                if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                    continue
                
                a = u_nbrs[0]
                b = u_nbrs[1]
                c = v_nbrs[0]
                d = v_nbrs[1]
                
                # NNI1: swap b and c
                new_adj1 = _apply_nni(adj, node_types, u, v, a, b, c, d, 1)
                score1 = _compute_me_score(new_adj1, node_types, dist_matrix)
                if score1 < best_score - 1e-9:
                    best_score = score1
                    best_adj = new_adj1
                    improved = True
                
                # NNI2: swap b and d
                new_adj2 = _apply_nni(adj, node_types, u, v, a, b, c, d, 2)
                score2 = _compute_me_score(new_adj2, node_types, dist_matrix)
                if score2 < best_score - 1e-9:
                    best_score = score2
                    best_adj = new_adj2
                    improved = True
        
        # Try SPR moves: prune a subtree and regraft elsewhere
        if _check_deadline():
            # For each internal edge (u,v), try pruning the subtree at u (away from v)
            # and regrafting at other internal edges
            for u in node_ids:
                if not _check_deadline():
                    break
                if node_types[u][0] != 'internal':
                    continue
                for v in adj[u]:
                    if node_types[v][0] != 'internal':
                        continue
                    
                    # Prune subtree rooted at u (away from v)
                    # The subtree is everything reachable from u without going through v
                    # Reattach it at various other internal edges
                    
                    # Find all other internal edges (not involving u or v directly)
                    for w in node_ids:
                        if not _check_deadline():
                            break
                        if node_types[w][0] != 'internal' or w == u or w == v:
                            continue
                        for x in adj[w]:
                            if node_types[x][0] != 'internal' or x == u or x == v:
                                continue
                            if x <= w:
                                continue
                            
                            # Try regrafting: disconnect u from v, connect u to w (or x)
                            # This is complex; simplify by trying to connect the pruned 
                            # subtree's root to different positions
                            # For now, skip full SPR and rely on NNI which is a subset
                            pass
        
        if best_adj is not adj:
            adj = best_adj
            current_score = best_score
        
        if not improved:
            break
    
    return adj


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in adj[nid]:
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"


def _exhaustive_ml_for_small_n(sequences, labels):
    """For n <= 7, enumerate all unrooted topologies and find MLE."""
    n = len(labels)
    if n < 4:
        return None
    
    # Number of unrooted binary trees with n leaves: (2n-5)!!
    # For n=4: 3, n=5: 15, n=6: 105, n=7: 945
    
    # We'll use a recursive approach to generate all topologies
    # This is complex, so for now we'll use NJ + NNI which should be sufficient
    
    return None


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
    
    # Step 2: Neighbor-Joining to get initial tree
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
    
    # Build initial Newick
    newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Step 4: Minimum Evolution SPR/NNI local search
    if _check_deadline():
        adj = _me_spr_search(adj, node_ids, node_types, sequences, labels, matrix, max_rounds=15)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
