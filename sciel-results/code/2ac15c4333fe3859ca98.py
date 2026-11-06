"""Phylogenetic topology recovery: NJ + quartet voting + exhaustive small-n + fast NNI local search."""
import math
import time
import numpy as np
from itertools import combinations, permutations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 258  # safety margin before 270s kill


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


def _bionj_distance(p, q, r, s, n_sites):
    """BionJ corrected distance for taxa p, q given reference taxa r, s."""
    # p, q, r, s are indices into the sequence list
    # d_pq = distance between p and q
    # The BionJ correction: d_pq = d_pq_raw + (1/2)(d_pr + d_qs - d_ps - d_qr) * correction
    # Actually BionJ is: d_pq = -1/2 * log(1 - 4/3 * (d_pq_raw + ...))
    # Simplified: use the standard BionJ formula
    # d_pq = d_pq_raw + (1/2) * (d_pr + d_qs - d_ps - d_qr) / (2 * n_sites) ... 
    # Let me use the proper BionJ formula from the paper:
    # For a site pattern, the BionJ distance is:
    # d_pq = -0.75 * ln(1 - 4/3 * (p_pq + (p_pr + p_qs - p_ps - p_qr)/2))
    # where p_xy is the proportion of differences between x and y
    pass


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


def _jc_likelihood_tree(adj, node_types, sequences, labels):
    """Compute ML log-likelihood of the tree given the sequences using Felsenstein's pruning.
    Uses optimized branch length estimation."""
    n_taxa = len(labels)
    if n_taxa < 2:
        return 0.0
    
    n_sites = len(sequences[labels[0]])
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build rooted structure
    node_ids = list(adj.keys())
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # Estimate branch lengths using NJ-style approach
    # For each edge, estimate length from the distance matrix
    # First compute pairwise distances
    dist = {}
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            d = _jc_distance(sequences[a], sequences[b])
            dist[(a, b)] = d
            dist[(b, a)] = d
    
    # Estimate branch lengths for each edge
    branch_len = {}
    for nid in node_ids:
        for nbr in adj[nid]:
            if nid > nbr:
                continue
            # Get leaves on each side of the edge
            leaves_a = _get_leaves_under(adj, node_types, nid, nbr)
            leaves_b = _get_leaves_under(adj, node_types, nbr, nid)
            
            if not leaves_a or not leaves_b:
                branch_len[(nid, nbr)] = 0.01
                continue
            
            # Estimate edge length as average of distances across the split
            # divided by 2 (since each pair's distance includes this edge once)
            cross_sum = 0.0
            count = 0
            for la in leaves_a:
                for lb in leaves_b:
                    cross_sum += dist.get((la, lb), 0.0)
                    count += 1
            
            if count > 0:
                # The edge contributes to each cross-pair distance
                # edge_len ≈ cross_sum / (2 * |A| * |B|)
                edge_len = cross_sum / (2.0 * len(leaves_a) * len(leaves_b))
                branch_len[(nid, nbr)] = max(0.001, min(1.0, edge_len))
            else:
                branch_len[(nid, nbr)] = 0.01
    
    # Precompute transition probabilities for each edge
    # p_same = 0.25 + 0.75 * exp(-4*L/3)
    # p_diff = (1 - exp(-4*L/3)) / 3
    edge_trans = {}
    for (u, v), L in branch_len.items():
        exp_term = math.exp(-4.0 * L / 3.0)
        p_same = 0.25 + 0.75 * exp_term
        p_diff = (1.0 - exp_term) / 3.0
        edge_trans[(u, v)] = (p_same, p_diff)
        edge_trans[(v, u)] = (p_same, p_diff)
    
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
                    p_same, p_diff = edge_trans.get((nid, child), (0.5, 0.25))
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


def _apply_nni(adj, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure."""
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


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    # Pick a leaf as root for display
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        children = []
        for nbr in adj[nid]:
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"


def _nni_local_search_fast(adj, node_types, sequences, labels, dist_matrix, max_rounds=50):
    """Perform NNI local search using fast quartet-based scoring.
    
    Uses the quartet agreement score as a fast proxy for ML likelihood.
    This is much faster than full ML scoring for each NNI move.
    """
    n_taxa = len(labels)
    optimal_splits = _quartet_score(dist_matrix, labels, n_taxa)
    
    # Precompute leaf sets for each edge (cached)
    def get_edge_split(u, v, adj, node_types):
        """Get the two leaf sets implied by edge (u,v)."""
        leaves_a = _get_leaves_under(adj, node_types, u, v)
        leaves_b = _get_leaves_under(adj, node_types, v, u)
        return frozenset(leaves_a), frozenset(leaves_b)
    
    current_score = _tree_quartet_agreement(adj, node_types, labels, optimal_splits)
    
    for round_num in range(max_rounds):
        if not _check_deadline():
            break
        
        improved = False
        best_score = current_score
        best_adj = adj
        
        node_ids = list(adj.keys())
        
        # Try all NNI moves
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
                new_adj1 = _apply_nni(adj, u, v, a, b, c, d, 1)
                score1 = _tree_quartet_agreement(new_adj1, node_types, labels, optimal_splits)
                if score1 > best_score:
                    best_score = score1
                    best_adj = new_adj1
                    improved = True
                
                # NNI2: swap b and d
                new_adj2 = _apply_nni(adj, u, v, a, b, c, d, 2)
                score2 = _tree_quartet_agreement(new_adj2, node_types, labels, optimal_splits)
                if score2 > best_score:
                    best_score = score2
                    best_adj = new_adj2
                    improved = True
        
        if best_adj is not adj:
            adj = best_adj
            current_score = best_score
        
        if not improved:
            break
    
    return adj


def _nni_local_search(adj, node_types, sequences, labels, max_rounds=20):
    """Perform NNI local search using ML scoring (fallback for larger trees)."""
    current_ll = _jc_likelihood_tree(adj, node_types, sequences, labels)
    
    for round_num in range(max_rounds):
        if not _check_deadline():
            break
        
        improved = False
        best_ll = current_ll
        best_adj = adj
        
        node_ids = list(adj.keys())
        
        # Try all NNI moves
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
                new_adj1 = _apply_nni(adj, u, v, a, b, c, d, 1)
                ll1 = _jc_likelihood_tree(new_adj1, node_types, sequences, labels)
                if ll1 > best_ll + 1e-9:
                    best_ll = ll1
                    best_adj = new_adj1
                    improved = True
                
                # NNI2: swap b and d
                new_adj2 = _apply_nni(adj, u, v, a, b, c, d, 2)
                ll2 = _jc_likelihood_tree(new_adj2, node_types, sequences, labels)
                if ll2 > best_ll + 1e-9:
                    best_ll = ll2
                    best_adj = new_adj2
                    improved = True
        
        if best_adj is not adj:
            adj = best_adj
            current_ll = best_ll
        
        if not improved:
            break
    
    return adj


def _quartet_score(dist_matrix, labels, n_taxa):
    """Compute total quartet support score for a given tree topology.
    
    For each quartet of 4 taxa, check which split the tree implies and
    compare with the distance-based best split. Returns number of quartets
    where the tree's split matches the distance-optimal split.
    """
    if n_taxa < 4:
        return 0
    
    # Build adjacency from labels (assumes we have a tree structure)
    # Instead, we'll use a different approach: count quartet agreements directly
    
    # For efficiency, precompute all quartet optimal splits
    optimal_splits = {}
    for a_idx in range(n_taxa):
        for b_idx in range(a_idx + 1, n_taxa):
            for c_idx in range(b_idx + 1, n_taxa):
                for d_idx in range(c_idx + 1, n_taxa):
                    a, b, c, d = labels[a_idx], labels[b_idx], labels[c_idx], labels[d_idx]
                    d_ab = dist_matrix.get((a, b), 0.0)
                    d_cd = dist_matrix.get((c, d), 0.0)
                    d_ac = dist_matrix.get((a, c), 0.0)
                    d_bd = dist_matrix.get((b, d), 0.0)
                    d_ad = dist_matrix.get((a, d), 0.0)
                    d_bc = dist_matrix.get((b, c), 0.0)
                    
                    sum1 = d_ab + d_cd
                    sum2 = d_ac + d_bd
                    sum3 = d_ad + d_bc
                    
                    min_sum = min(sum1, sum2, sum3)
                    if sum1 == min_sum:
                        optimal_splits[(a, b, c, d)] = (frozenset([a, b]), frozenset([c, d]))
                    elif sum2 == min_sum:
                        optimal_splits[(a, b, c, d)] = (frozenset([a, c]), frozenset([b, d]))
                    else:
                        optimal_splits[(a, b, c, d)] = (frozenset([a, d]), frozenset([b, c]))
    
    return optimal_splits


def _tree_quartet_agreement(adj, node_types, labels, optimal_splits):
    """Count how many quartets the tree agrees with optimal splits."""
    n_taxa = len(labels)
    if n_taxa < 4:
        return 0
    
    # For each edge, determine which leaves are on each side
    # A split (A|B) is implied by edge if removing edge separates leaves into A and B
    
    # Build leaf-to-node mapping
    leaf_nodes = {}
    for nid in adj:
        if node_types[nid][0] == 'leaf':
            leaf_nodes[node_types[nid][1]] = nid
    
    # For each edge, compute the split it implies
    edge_splits = {}
    for u in adj:
        for v in adj[u]:
            if u > v:
                continue
            # Get leaves on each side
            leaves_a = _get_leaves_under(adj, node_types, u, v)
            leaves_b = _get_leaves_under(adj, node_types, v, u)
            if len(leaves_a) >= 2 and len(leaves_b) >= 2:
                split = (frozenset(leaves_a), frozenset(leaves_b))
                edge_splits[(u, v)] = split
    
    # Count agreements
    agreement = 0
    total_quartets = 0
    for (a, b, c, d), (opt_a, opt_b) in optimal_splits.items():
        total_quartets += 1
        # Check if the tree has this split
        for split in edge_splits.values():
            if split == (opt_a, opt_b) or split == (opt_b, opt_a):
                agreement += 1
                break
    
    return agreement


def _exhaustive_small_n(dist_matrix, labels, n_taxa):
    """Exhaustive search over all unrooted tree topologies for small n (<=7).
    
    Uses quartet scoring as a fast proxy for ML.
    Returns the best Newick string.
    """
    if n_taxa < 4:
        return "(" + ",".join(labels) + ");"
    
    optimal_splits = _quartet_score(dist_matrix, labels, n_taxa)
    
    # For n=4: 3 possible topologies
    # For n=5: 15 possible topologies
    # For n=6: 105 possible topologies
    # For n=7: 945 possible topologies
    
    # Generate all possible unrooted tree topologies using a recursive approach
    # Represent trees as adjacency structures with integer node IDs
    
    best_newick = None
    best_score = -1
    
    if n_taxa == 4:
        # 3 possible quartet topologies
        a, b, c, d = labels[0], labels[1], labels[2], labels[3]
        candidates = [
            f"(({a},{b}),({c},{d}));",
            f"(({a},{c}),({b},{d}));",
            f"(({a},{d}),({b},{c}));",
        ]
        for newick in candidates:
            tree = _parse_newick(newick)
            adj, node_ids, node_types = _build_adjacency(tree)
            score = _tree_quartet_agreement(adj, node_types, labels, optimal_splits)
            if score > best_score:
                best_score = score
                best_newick = newick
    
    elif n_taxa == 5:
        # 15 possible unrooted binary trees on 5 leaves
        # Generate by adding one leaf to each of the 3 quartet trees at each edge
        a, b, c, d, e = labels
        # All 15 topologies can be enumerated
        # Use a systematic approach: fix a central structure
        quartet_trees = [
            [a, b, c, d],  # will add e
        ]
        
        # Generate all 15 unrooted binary trees on 5 labeled leaves
        # A unrooted binary tree on 5 leaves has 3 internal nodes
        # We can enumerate by considering which pairs are "sisters"
        
        # Method: start with all ways to pick a pair of leaves to be sisters,
        # then the remaining 3 form a subtree attached to the internal node
        # Actually, let's use a different enumeration:
        # All unrooted binary trees on 5 leaves correspond to ways of
        # partitioning into a 2-leaf clade and a 3-leaf clade, then
        # the 3-leaf clade has a specific structure
        
        # Simpler: enumerate all 15 by considering the 5 choices for
        # which leaf is "apical" and the 3 choices for how the other 4 are arranged
        # But that overcounts. Let's just use a known list.
        
        # The 15 unrooted binary trees on 5 leaves can be generated by:
        # Pick a 2-leaf clade (10 choices), then the remaining 3 leaves
        # form a 3-leaf tree (1 way up to symmetry) attached to the 
        # internal node of the 2-leaf clade. But this gives 10, not 15.
        
        # Actually: for 5 leaves, there are 15 = (5-1)!/2 * ... 
        # Let me just enumerate systematically.
        
        # A unrooted binary tree on n leaves has (2n-5)!! edges... no.
        # Number of unrooted binary trees on n labeled leaves: (2n-5)!!
        # n=4: 3!! = 3
        # n=5: 5!! = 15
        # n=6: 7!! = 105
        # n=7: 9!! = 945
        
        # Generate all 15 for n=5:
        # Fix leaf a as reference. The tree can be described by
        # which other leaf is its "sister" (4 choices), and then
        # the remaining 3 leaves form a 3-leaf subtree (1 topology).
        # But the 3-leaf subtree can be attached in different ways.
        
        # Let me use a constructive approach:
        # Start with a 4-leaf tree (3 choices for topology), add 5th leaf
        # to each of the 5 edges of the 4-leaf tree.
        # 3 * 5 = 15. 
        
        all_trees = []
        quartet_topos = [
            (a, b, c, d),  # (a,b)|(c,d)
            (a, c, b, d),  # (a,c)|(b,d) 
            (a, d, b, c),  # (a,d)|(b,c)
        ]
        
        # For each quartet topology, add e to each of the 5 edges
        # Edge types in a 4-leaf tree: 2 internal edges + ... 
        # Actually a 4-leaf unrooted tree has 5 edges: 4 leaf edges + 1 internal edge
        # Adding to a leaf edge creates a new "sister" relationship
        # Adding to the internal edge creates a different topology
        
        # Let me just enumerate all 15 explicitly for the 5 labels
        # using the fact that they correspond to all ways to choose
        # a "caterpillar" or "balanced" structure
        
        # Explicit enumeration of 15 unrooted binary trees on 5 leaves {a,b,c,d,e}:
        # Type 1: balanced (2+3 split)
        #   Choose which 2 are sisters: C(5,2)=10 ways, but each tree counted once
        #   Actually for a 2+3 split, the 3-leaf part has 1 topology
        #   So 10 trees of this type? No, because the attachment point matters.
        
        # I'll use a different strategy: generate all by recursive leaf addition
        def gen_trees(leaf_list):
            """Generate all unrooted binary tree topologies as adjacency dicts."""
            n = len(leaf_list)
            if n == 3:
                # Only one unrooted binary tree on 3 leaves (star, but binary means one internal node with 3 children... wait)
                # Actually for 3 leaves, the unrooted binary tree is just a 3-star (one internal node)
                # But that's not binary in the usual sense. For our purposes, 3-leaf tree is unique.
                adj = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
                node_types = {0: ('internal', None), 1: ('leaf', leaf_list[0]), 2: ('leaf', leaf_list[1]), 3: ('leaf', leaf_list[2])}
                return [(adj, node_types)]
            
            if n == 4:
                # 3 possible topologies
                l0, l1, l2, l3 = leaf_list
                results = []
                # (l0,l1)|(l2,l3)
                adj1 = {0: {1, 2}, 1: {0, 3, 4}, 2: {0, 5, 6}, 3: {1}, 4: {1}, 5: {2}, 6: {2}}
                nt1 = {0: ('internal', None), 1: ('internal', None), 2: ('internal', None), 
                       3: ('leaf', l0), 4: ('leaf', l1), 5: ('leaf', l2), 6: ('leaf', l3)}
                results.append((adj1, nt1))
                # (l0,l2)|(l1,l3)
                adj2 = {0: {1, 2}, 1: {0, 3, 4}, 2: {0, 5, 6}, 3: {1}, 4: {1}, 5: {2}, 6: {2}}
                nt2 = {0: ('internal', None), 1: ('internal', None), 2: ('internal', None),
                       3: ('leaf', l0), 4: ('leaf', l2), 5: ('leaf', l1), 6: ('leaf', l3)}
                results.append((adj2, nt2))
                # (l0,l3)|(l1,l2)
                adj3 = {0: {1, 2}, 1: {0, 3, 4}, 2: {0, 5, 6}, 3: {1}, 4: {1}, 5: {2}, 6: {2}}
                nt3 = {0: ('internal', None), 1: ('internal', None), 2: ('internal', None),
                       3: ('leaf', l0), 4: ('leaf', l3), 5: ('leaf', l1), 6: ('leaf', l2)}
                results.append((adj3, nt3))
                return results
            
            # For n > 4, use leaf addition
            # Take all trees on n-1 leaves, add the nth leaf to each edge
            prev_trees = gen_trees(leaf_list[:-1])
            new_leaf = leaf_list[-1]
            results = []
            for adj, nt in prev_trees:
                # Find all edges
                edges = set()
                for u in adj:
                    for v in adj[u]:
                        if u < v:
                            edges.add((u, v))
                
                for (u, v) in edges:
                    # Subdivide edge (u,v) with new internal node
                    new_id = max(adj.keys()) + 1
                    new_adj = {k: set(vv) for k, vv in adj.items()}
                    new_adj[u].discard(v)
                    new_adj[v].discard(u)
                    new_adj[u].add(new_id)
                    new_adj[v].add(new_id)
                    new_adj[new_id] = {u, v, new_id + 1}
                    new_adj[new_id + 1] = {new_id}
                    new_nt = dict(nt)
                    new_nt[new_id] = ('internal', None)
                    new_nt[new_id + 1] = ('leaf', new_leaf)
                    results.append((new_adj, new_nt))
            
            return results
        
        all_trees = gen_trees(labels)
        
        for adj, node_types in all_trees:
            score = _tree_quartet_agreement(adj, node_types, labels, optimal_splits)
            if score > best_score:
                best_score = score
                best_newick = _adj_to_newick(adj, node_types)
    
    elif n_taxa == 6:
        # 105 possible topologies - use NJ as starting point and do exhaustive NNI
        # Actually, let's just use the fast NNI local search which should be fast enough
        # for 105 topologies
        pass
    
    elif n_taxa == 7:
        # 945 possible topologies - use NJ + NNI
        pass
    
    if best_newick is not None:
        return best_newick
    
    return None


def _fast_nni_score(dist_matrix, labels, n_taxa):
    """Compute a fast quartet-based score for NNI moves.
    
    For an NNI move on edge (u,v) with neighbors a,b on u and c,d on v,
    the change in quartet score can be computed by looking at all quartets
    that include leaves from both sides of the edge.
    """
    pass


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
    
    # Step 1b: For small n (<=7), try exhaustive search
    if n_taxa <= 7:
        exhaustive_result = _exhaustive_small_n(matrix, labels, n_taxa)
        if exhaustive_result is not None:
            # Validate and return if exhaustive search succeeded
            for label in labels:
                if label not in exhaustive_result:
                    break
            else:
                return exhaustive_result
    
    # Step 2: Neighbor-Joining to get initial tree
    nodes = list(labels)
    trees = {name: name for name in labels}
    
    while len(nodes) > 3:
        if not _check_deadline():
            break
        n = len(nodes)
        total = {a: sum(matrix.get((a, b), 0.0) for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix.get((a, b), 0.0) - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        if best is None:
            break
        
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix.get((a, x), 0.0) + matrix.get((b, x), 0.0) - matrix.get((a, b), 0.0))
        nodes = rest + [new]
        trees[new] = joined
    
    # Build initial Newick
    newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Step 4: NNI local search - use fast quartet-based scoring for speed
    if _check_deadline():
        if n_taxa <= 20:
            # Use fast quartet-based NNI for smaller trees
            adj = _nni_local_search_fast(adj, node_types, sequences, labels, matrix, max_rounds=50)
        else:
            # Use ML-based NNI for larger trees (fewer moves needed)
            adj = _nni_local_search(adj, node_types, sequences, labels, max_rounds=5)
    
    # Step 5: If time remains, do a second pass with ML scoring for refinement
    if _check_deadline() and n_taxa <= 15:
        adj = _nni_local_search(adj, node_types, sequences, labels, max_rounds=3)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
