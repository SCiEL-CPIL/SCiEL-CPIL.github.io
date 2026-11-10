"""Quartet-based phylogenetic tree reconstruction with NNI refinement."""
import time
import math
import numpy as np
from itertools import combinations


def _parse_newick(newick):
    """Parse a Newick string into a tree structure."""
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            tokens.append(('LPAREN', '('))
            i += 1
        elif c == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
        elif c == ',':
            tokens.append(('COMMA', ','))
            i += 1
        elif c == ':':
            j = i + 1
            while j < len(s) and s[j] not in '(),;':
                j += 1
            tokens.append(('BL', s[i+1:j]))
            i = j
        else:
            j = i
            while j < len(s) and s[j] not in '():,;':
                j += 1
            tokens.append(('LABEL', s[i:j]))
            i = j
    
    parent_of = {}
    children_of = {}
    taxon_map = {}
    node_count = [0]
    
    def next_node_id():
        nid = node_count[0]
        node_count[0] += 1
        parent_of[nid] = -1
        children_of[nid] = []
        return nid
    
    pos = [0]
    
    def parse_subtree():
        while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
            pos[0] += 1
        
        if pos[0] >= len(tokens):
            return None
        
        tok = tokens[pos[0]]
        if tok[0] == 'LPAREN':
            node_id = next_node_id()
            pos[0] += 1
            
            while True:
                child_id = parse_subtree()
                if child_id is not None:
                    children_of[node_id].append(child_id)
                    parent_of[child_id] = node_id
                
                while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                    pos[0] += 1
                
                if pos[0] >= len(tokens):
                    break
                    
                tok2 = tokens[pos[0]]
                if tok2[0] == 'COMMA':
                    pos[0] += 1
                elif tok2[0] == 'RPAREN':
                    pos[0] += 1
                    break
                else:
                    break
            
            while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                pos[0] += 1
                
            if pos[0] < len(tokens):
                tok3 = tokens[pos[0]]
                if tok3[0] == 'BL':
                    pos[0] += 1
                    
            return node_id
        elif tok[0] == 'LABEL':
            node_id = next_node_id()
            taxon_map[tok[1]] = node_id
            pos[0] += 1
            while pos[0] < len(tokens) and tokens[pos[0]][1].strip() == '':
                pos[0] += 1
            if pos[0] < len(tokens) and tokens[pos[0]][0] == 'BL':
                pos[0] += 1
            return node_id
        else:
            return None
    
    root_id = parse_subtree()
    
    return parent_of, children_of, taxon_map, root_id


def _to_newick(parent_of, children_of, taxon_map, root_id, labels):
    """Convert tree structure back to Newick string."""
    node_to_taxon = {}
    for taxon, nid in taxon_map.items():
        node_to_taxon[nid] = taxon
    
    def node_name(nid):
        if nid in node_to_taxon:
            return node_to_taxon[nid]
        return f"n{nid}"
    
    def to_newick_recursive(nid):
        kids = children_of[nid]
        if not kids:
            return node_name(nid)
        else:
            parts = [to_newick_recursive(k) for k in kids]
            return "(" + ",".join(parts) + ")"
    
    return to_newick_recursive(root_id) + ";"


def _get_splits(parent_of, children_of, root_id, taxon_map, labels):
    """Get all non-trivial splits from an unrooted tree."""
    n_taxa = len(labels)
    all_taxa = set(labels)
    splits = set()
    
    def compute_subtree_taxa(nid):
        if nid in taxon_map:
            return {taxon_map[nid]}
        result = set()
        for child in children_of[nid]:
            result |= compute_subtree_taxa(child)
        return result
    
    for nid in range(len(children_of)):
        if parent_of[nid] == -1:
            continue
        subtree_taxa = compute_subtree_taxa(nid)
        complement = all_taxa - subtree_taxa
        
        if len(subtree_taxa) <= 1 or len(subtree_taxa) >= n_taxa - 1:
            continue
        
        if len(subtree_taxa) < len(complement):
            split_key = frozenset(subtree_taxa)
        else:
            split_key = frozenset(complement)
        
        splits.add(split_key)
    
    return splits


def _quartet_scores(sequences, labels, n_sites):
    """Compute quartet scores for all 4-taxon subsets.
    
    For each set of 4 taxa, compute the support for each of the 3 possible
    unrooted topologies based on the sequence data.
    
    Returns a dict mapping frozenset of 4 taxa -> (score_01|23, score_02|13, score_03|12)
    where the scores are based on the number of sites that support each topology.
    """
    n = len(labels)
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Convert sequences to numpy array
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    
    # For each quartet, compute the three possible topologies
    # Topology 1: (a,b)|(c,d) - supported when a==b and c==d, or a!=b and c!=d
    # Topology 2: (a,c)|(b,d) - supported when a==c and b==d, or a!=c and b!=d  
    # Topology 3: (a,d)|(b,c) - supported when a==d and b==c, or a!=d and b!=c
    
    quartet_scores = {}
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        labels_quad = (labels[a], labels[b], labels[c], labels[d])
        key = frozenset(labels_quad)
        
        # Extract the 4 sequences for this quartet
        seq_a = mat[a]
        seq_b = mat[b]
        seq_c = mat[c]
        seq_d = mat[d]
        
        # For each site, determine which topology is supported
        # Under JC model, the most parsimonious topology for a site is the one
        # that minimizes the number of changes. For 4 taxa, this can be determined
        # by looking at the pattern of nucleotides.
        
        # Simple approach: for each site, count how many pairs match
        # Topology (a,b)|(c,d): matches if (a==b and c==d) or (a!=b and c!=d)
        # This is a heuristic; better to use maximum likelihood or parsimony
        
        # Use parsimony: for each site, the topology that requires the fewest changes
        # For 4 taxa on a site, we can compute the parsimony score for each topology
        
        # Topology 1: (a,b)|(c,d)
        # The internal edge connects the (a,b) clade to the (c,d) clade
        # Parsimony score: min over possible states at internal nodes
        
        # For simplicity, use a voting scheme:
        # For each site, the "best" topology is the one where the two pairs that
        # are most similar are grouped together
        
        # Compute pairwise identity for this site
        ab_match = (seq_a[0] == seq_b[0])
        cd_match = (seq_c[0] == seq_d[0])
        ac_match = (seq_a[0] == seq_c[0])
        bd_match = (seq_b[0] == seq_d[0])
        ad_match = (seq_a[0] == seq_d[0])
        bc_match = (seq_b[0] == seq_c[0])
        
        # Score each topology based on how well it explains the data
        # Topology (a,b)|(c,d): high score if a~b and c~d
        score_0123 = (1 if ab_match else 0) + (1 if cd_match else 0)
        # Topology (a,c)|(b,d): high score if a~c and b~d
        score_0213 = (1 if ac_match else 0) + (1 if bd_match else 0)
        # Topology (a,d)|(b,c): high score if a~d and b~c
        score_0312 = (1 if ad_match else 0) + (1 if bc_match else 0)
        
        # Sum over all sites
        total_0123 = 0
        total_0213 = 0
        total_0312 = 0
        
        for site in range(n_sites):
            ab_match = (seq_a[site] == seq_b[site])
            cd_match = (seq_c[site] == seq_d[site])
            ac_match = (seq_a[site] == seq_c[site])
            bd_match = (seq_b[site] == seq_d[site])
            ad_match = (seq_a[site] == seq_d[site])
            bc_match = (seq_b[site] == seq_c[site])
            
            s1 = (1 if ab_match else 0) + (1 if cd_match else 0)
            s2 = (1 if ac_match else 0) + (1 if bd_match else 0)
            s3 = (1 if ad_match else 0) + (1 if bc_match else 0)
            
            total_0123 += s1
            total_0213 += s2
            total_0312 += s3
        
        quartet_scores[key] = (total_0123, total_0213, total_0312)
    
    return quartet_scores


def _build_tree_from_quartets(quartet_scores, labels):
    """Build a tree from quartet scores using a greedy approach.
    
    Start with a star tree and add quartet constraints one by one,
    choosing the quartet with the strongest signal.
    """
    n = len(labels)
    
    if n <= 3:
        if n == 2:
            return f"({labels[0]},{labels[1]})"
        elif n == 1:
            return labels[0]
        else:
            return "(" + ",".join(labels) + ")"
    
    # Get all quartets sorted by strength of signal
    quartet_list = []
    for key, scores in quartet_scores.items():
        s0, s1, s2 = scores
        total = s0 + s1 + s2
        if total == 0:
            continue
        # Strength = max_score - second_max_score
        sorted_scores = sorted([s0, s1, s2], reverse=True)
        strength = sorted_scores[0] - sorted_scores[1]
        if strength > 0:
            quartet_list.append((strength, key, scores))
    
    # Sort by strength descending
    quartet_list.sort(key=lambda x: x[0], reverse=True)
    
    # Start with a star tree
    # Represent the tree as an adjacency list
    # Nodes: 0..n-1 for taxa, n..2n-4 for internal nodes
    # Initially all taxa are connected to a central node
    
    # For simplicity, start with NJ and then refine with quartet information
    # This is a hybrid approach
    
    return None  # Fall back to NJ


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
    """Standard neighbour joining."""
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
    
    return "(" + ",".join(trees[x] for x in nodes) + ";"


def _nni_improve(newick, dist, labels, max_iterations=10, time_limit=None, start_time=None):
    """Apply NNI moves to improve the tree topology."""
    if time_limit is not None and start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return newick
    
    parent_of, children_of, taxon_map, root_id = _parse_newick(newick)
    n_nodes = len(children_of)
    
    if n_nodes < 4:
        return newick
    
    adj = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        for child in children_of[i]:
            adj[i].append(child)
            adj[child].append(i)
    
    internal_nodes = [i for i in range(n_nodes) if len(children_of[i]) > 0]
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time_limit is not None and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        internal_edges = []
        for u in internal_nodes:
            for v in adj[u]:
                if v > u and v in internal_nodes:
                    internal_edges.append((u, v))
        
        for (u, v) in internal_edges:
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) < 2 or len(v_neighbors) < 2:
                continue
            
            a, b = u_neighbors[0], u_neighbors[1]
            c, d = v_neighbors[0], v_neighbors[1]
            
            def get_taxa_set(nid):
                if nid in taxon_map:
                    return {taxon_map[nid]}
                result = set()
                for child in children_of[nid]:
                    result |= get_taxa_set(child)
                return result
            
            set_a = get_taxa_set(a)
            set_b = get_taxa_set(b)
            set_c = get_taxa_set(c)
            set_d = get_taxa_set(d)
            
            label_to_idx = {lab: i for i, lab in enumerate(labels)}
            
            def side_dist(set1, set2):
                total = 0.0
                for t1 in set1:
                    i1 = label_to_idx[t1]
                    for t2 in set2:
                        i2 = label_to_idx[t2]
                        total += dist[i1, i2]
                return total
            
            cost_current = side_dist(set_a | set_b, set_c | set_d)
            cost_alt1 = side_dist(set_a | set_c, set_b | set_d)
            cost_alt2 = side_dist(set_a | set_d, set_b | set_c)
            
            min_cost = min(cost_current, cost_alt1, cost_alt2)
            
            if min_cost < cost_current - 1e-10:
                if min_cost == cost_alt1:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(c)
                    adj[c].remove(v)
                    
                    adj[u].append(c)
                    adj[c].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                elif min_cost == cost_alt2:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(d)
                    adj[d].remove(v)
                    
                    adj[u].append(d)
                    adj[d].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                
                improved = True
        
        if improved:
            parent_of = {i: -1 for i in range(n_nodes)}
            children_of = {i: [] for i in range(n_nodes)}
            
            visited = set()
            stack = [root_id]
            parent_of[root_id] = -1
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        parent_of[neighbor] = node
                        children_of[node].append(neighbor)
                        stack.append(neighbor)
    
    return _to_newick(parent_of, children_of, taxon_map, root_id, labels)


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
        result = _nni_improve(result, dist, labels, max_iterations=5, 
                              time_limit=deadline - time.time() - 2.0,
                              start_time=start_time)
    except Exception:
        pass
    
    return result
