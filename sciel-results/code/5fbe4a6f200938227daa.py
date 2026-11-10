<<<<<<< SEARCH
"""Reconstruct phylogenetic topology using Neighbor Joining with Minimum Evolution refinement."""
import numpy as np
import time
import math

_START_TIME = None
_DEADLINE = 250.0  # seconds, safety margin before 270s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    
    # Convert to dictionary for flexible node IDs
    d_dict = {}
    for i in range(n):
        for j in range(n):
            d_dict[(labels[i], labels[j])] = d[i, j]
    return d_dict, labels

def _nj_newick_dict(d_dict, labels):
    """NJ using dictionary for distances. Returns newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    # Copy distance matrix to dictionary
    dist = {k: v for k, v in d_dict.items()}
    
    # Helper to get distance
    def get_dist(u, v):
        if (u, v) in dist:
            return dist[(u, v)]
        if (v, u) in dist:
            return dist[(v, u)]
        return 0.0

    # Helper to set distance
    def set_dist(u, v, val):
        dist[(u, v)] = val
        dist[(v, u)] = val

    active = list(labels)
    trees = {label: label for label in labels}
    
    next_internal_id = 0
    internal_names = []
    
    while len(active) > 3:
        n_act = len(active)
        
        # Compute row sums r[i]
        r = {}
        for i in active:
            r[i] = sum(get_dist(i, k) for k in active if k != i)
        
        # Find min Q
        min_q = float('inf')
        min_i = None
        min_j = None
        for i_idx in range(n_act):
            i = active[i_idx]
            for j_idx in range(i_idx + 1, n_act):
                j = active[j_idx]
                d_ij = get_dist(i, j)
                q = (n_act - 2) * d_ij - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        i = min_i
        j = min_j
        d_ij = get_dist(i, j)
        
        a = f"n{next_internal_id}"
        next_internal_id += 1
        
        # Update distances
        for k in active:
            if k == i or k == j:
                continue
            d_ik = get_dist(i, k)
            d_jk = get_dist(j, k)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            set_dist(a, k, d_ak)
        
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"

def _get_splits(newick):
    """Extract all non-trivial splits from a Newick string.
    Returns a list of frozensets of taxon labels.
    """
    # Simple parser to get splits
    # We need to map internal nodes to their leaf sets
    nodes = {}
    edges = []
    node_counter = [0]
    
    def parse(s, pos):
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
                    break
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',);':
                    pos += 1
            node_id = node_counter[0]
            node_counter[0] += 1
            nodes[node_id] = children
            for c in children:
                edges.append((node_id, c))
            return node_id, pos
        else:
            name = ""
            while pos < len(s) and s[pos] not in ',);:':
                name += s[pos]
                pos += 1
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',);':
                    pos += 1
            node_id = node_counter[0]
            node_counter[0] += 1
            nodes[node_id] = []
            return node_id, pos
    
    root, _ = parse(newick, 0)
    
    # Compute leaf sets for each node
    leaf_sets = {}
    def compute_leaves(nid):
        if nid in leaf_sets:
            return leaf_sets[nid]
        children = nodes[nid]
        if not children:
            # Leaf node: name is not stored in nodes, we need to recover it
            # This is a flaw in the parser above. Let's fix it by storing names.
            # Actually, the parser above doesn't store names for leaves.
            # Let's rewrite the parser to store names.
            pass
    
    # The parser above is insufficient. Let's use a more robust method:
    # We can just use the distance matrix to compute RF distance directly?
    # No, we need splits.
    
    # Let's re-parse with names.
    nodes = {}
    names = {}
    node_counter = [0]
    
    def parse2(s, pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child_id, pos = parse2(s, pos)
                children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',);':
                    pos += 1
            node_id = node_counter[0]
            node_counter[0] += 1
            nodes[node_id] = children
            return node_id, pos
        else:
            name = ""
            while pos < len(s) and s[pos] not in ',);:':
                name += s[pos]
                pos += 1
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',);':
                    pos += 1
            node_id = node_counter[0]
            node_counter[0] += 1
            nodes[node_id] = []
            names[node_id] = name
            return node_id, pos
    
    root, _ = parse2(newick, 0)
    
    leaf_sets = {}
    def get_leaf_set(nid):
        if nid in leaf_sets:
            return leaf_sets[nid]
        children = nodes[nid]
        if not children:
            s = frozenset([names[nid]])
        else:
            s = frozenset().union(*[get_leaf_set(c) for c in children])
        leaf_sets[nid] = s
        return s
    
    all_leaves = get_leaf_set(root)
    splits = set()
    for nid in nodes:
        s = get_leaf_set(nid)
        if 0 < len(s) < len(all_leaves):
            # Canonicalize split: keep the one with smaller min element
            # Or just store both? RF distance uses sets of splits.
            # Standard: split {S, T} is represented by S if min(S) < min(T)
            # But we just need to match sets.
            splits.add(s)
            
    return list(splits)

def _rf_distance(splits1, splits2):
    """Calculate normalized Robinson-Foulds distance."""
    set1 = set(splits1)
    set2 = set(splits2)
    # Also need to add the complementary splits?
    # No, usually we just compare the set of splits.
    # However, a split {S, T} is the same as {T, S}.
    # The parser returns S. If the tree has {T, S} where T is smaller, it might return T.
    # To be safe, we should normalize.
    
    # Let's normalize: for each split s, if min(s) > min(complement), swap?
    # We don't have the complement easily here unless we pass all_leaves.
    # But since we are comparing two trees with the same leaves, we can just
    # ensure that for each split, we store the one with the lexicographically smaller min.
    
    def normalize_split(s, all_leaves):
        comp = all_leaves - s
        if min(s) < min(comp):
            return s
        else:
            return comp
    
    # We need all_leaves. Let's assume the input splits are already from the same leaf set.
    # We can infer all_leaves from the union of splits? No.
    # Let's just use the sets as is. The parser returns a specific side.
    # If Tree 1 returns {A, B} and Tree 2 returns {C, D} for the same cut, they won't match.
    # So we MUST normalize.
    
    # Since we don't have all_leaves passed in, we can't normalize perfectly here.
    # But we can modify _get_splits to return normalized splits if we pass all_leaves.
    # Let's change _get_splits to return (splits, all_leaves).
    
    # For now, this function is a stub. We will implement RF distance inside the refine loop
    # where we have access to all_leaves.
    pass

def _nni_refine_dict(newick, d_dict, labels, max_rounds=10):
    """NNI refinement using dictionary distance matrix."""
    if len(labels) < 4:
        return newick
    
    all_leaves = frozenset(labels)
    
    def get_splits_normalized(newick_str):
        # Reuse logic from _get_splits but normalize
        nodes = {}
        names = {}
        node_counter = [0]
        
        def parse2(s, pos):
            if s[pos] == '(':
                pos += 1
                children = []
                while True:
                    child_id, pos = parse2(s, pos)
                    children.append(child_id)
                    if pos < len(s) and s[pos] == ',':
                        pos += 1
                    elif pos < len(s) and s[pos] == ')':
                        pos += 1
                        break
                    else:
                        break
                if pos < len(s) and s[pos] == ':':
                    pos += 1
                    while pos < len(s) and s[pos] not in ',);':
                        pos += 1
                node_id = node_counter[0]
                node_counter[0] += 1
                nodes[node_id] = children
                return node_id, pos
            else:
                name = ""
                while pos < len(s) and s[pos] not in ',);:':
                    name += s[pos]
                    pos += 1
                if pos < len(s) and s[pos] == ':':
                    pos += 1
                    while pos < len(s) and s[pos] not in ',);':
                        pos += 1
                node_id = node_counter[0]
                node_counter[0] += 1
                nodes[node_id] = []
                names[node_id] = name
                return node_id, pos
        
        root, _ = parse2(newick_str, 0)
        
        leaf_sets = {}
        def get_leaf_set(nid):
            if nid in leaf_sets:
                return leaf_sets[nid]
            children = nodes[nid]
            if not children:
                s = frozenset([names[nid]])
            else:
                s = frozenset().union(*[get_leaf_set(c) for c in children])
            leaf_sets[nid] = s
            return s
        
        splits = set()
        for nid in nodes:
            s = get_leaf_set(nid)
            if 0 < len(s) < len(all_leaves):
                comp = all_leaves - s
                # Normalize: keep the set with smaller min
                if min(s) < min(comp):
                    splits.add(s)
                else:
                    splits.add(comp)
        return splits
    
    current_newick = newick
    current_splits = get_splits_normalized(current_newick)
    
    for _ in range(max_rounds):
        if _time_left() < 2.0:
            break
        
        improved = False
        # Parse current newick to find internal edges
        # We need the tree structure to find NNI moves.
        # This requires a robust parser that returns the tree structure.
        
        # Given time constraints, I will implement a simpler heuristic:
        # Just return the NJ tree. NNI is complex to implement correctly from scratch in this format.
        # However, NJ is a good baseline.
        
        # To improve score, we can try a few random NNI-like swaps if we had a tree structure.
        # But without a reliable tree structure parser that maps to the newick string parts,
        # it's risky.
        
        # Let's stick to NJ for now to ensure it runs.
        break
        
    return current_newick

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    d_dict, _ = _jc_distance_matrix(sequences)
    
    # Get NJ tree
    newick = _nj_newick_dict(d_dict, labels)
    
    # Refinement is disabled for now to ensure stability and speed
    # The NJ tree is a solid baseline.
    
    return newick
=======
"""Reconstruct phylogenetic topology using Neighbor Joining with NNI/ME refinement."""
import numpy as np
import time
import math

_START_TIME = None
_DEADLINE = 250.0  # seconds, safety margin before 270s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    
    # Convert to dictionary for flexible node IDs
    d_dict = {}
    for i in range(n):
        for j in range(n):
            d_dict[(labels[i], labels[j])] = d[i, j]
    return d_dict, labels

def _nj_tree_struct(d_dict, labels):
    """NJ using dictionary for distances. Returns tree structure as nested tuples."""
    n = len(labels)
    if n == 1:
        return (labels[0],)
    if n == 2:
        return ((labels[0],), (labels[1],))
    if n == 3:
        return ((labels[0],), (labels[1],), (labels[2],))

    dist = {k: v for k, v in d_dict.items()}
    
    def get_dist(u, v):
        if (u, v) in dist:
            return dist[(u, v)]
        if (v, u) in dist:
            return dist[(v, u)]
        return 0.0

    def set_dist(u, v, val):
        dist[(u, v)] = val
        dist[(v, u)] = val

    active = list(labels)
    # trees maps node id to its structure (tuple of children)
    trees = {label: (label,) for label in labels}
    
    next_internal_id = 0
    
    while len(active) > 3:
        n_act = len(active)
        
        r = {}
        for i in active:
            r[i] = sum(get_dist(i, k) for k in active if k != i)
        
        min_q = float('inf')
        min_i = None
        min_j = None
        for i_idx in range(n_act):
            i = active[i_idx]
            for j_idx in range(i_idx + 1, n_act):
                j = active[j_idx]
                d_ij = get_dist(i, j)
                q = (n_act - 2) * d_ij - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        i = min_i
        j = min_j
        d_ij = get_dist(i, j)
        
        a = f"n{next_internal_id}"
        next_internal_id += 1
        
        for k in active:
            if k == i or k == j:
                continue
            d_ik = get_dist(i, k)
            d_jk = get_dist(j, k)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            set_dist(a, k, d_ak)
        
        trees[a] = (trees[i], trees[j])
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return (trees[a], trees[b], trees[c])

def _tree_to_newick(tree_struct):
    """Convert tree structure (nested tuples) to Newick string."""
    if isinstance(tree_struct, str):
        return tree_struct
    if len(tree_struct) == 1:
        return _tree_to_newick(tree_struct[0])
    elif len(tree_struct) == 2:
        return "(" + _tree_to_newick(tree_struct[0]) + "," + _tree_to_newick(tree_struct[1]) + ")"
    else:
        parts = [_tree_to_newick(c) for c in tree_struct]
        return "(" + ",".join(parts) + ")"

def _newick_to_tree(newick):
    """Parse Newick string to tree structure (nested tuples)."""
    pos = [0]
    
    def parse():
        if pos[0] >= len(newick):
            return None
        if newick[pos[0]] == '(':
            pos[0] += 1
            children = []
            while True:
                child = parse()
                if child is None:
                    break
                children.append(child)
                if pos[0] < len(newick) and newick[pos[0]] == ',':
                    pos[0] += 1
                    continue
                elif pos[0] < len(newick) and newick[pos[0]] == ')':
                    pos[0] += 1
                    break
                else:
                    break
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(newick) and newick[pos[0]] not in ',);':
                    pos[0] += 1
            return tuple(children)
        else:
            name = ""
            while pos[0] < len(newick) and newick[pos[0]] not in ',);:':
                name += newick[pos[0]]
                pos[0] += 1
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(newick) and newick[pos[0]] not in ',);':
                    pos[0] += 1
            return (name,)
    
    return parse()

def _get_all_leaves(tree_struct):
    """Get all leaf labels from tree structure."""
    if isinstance(tree_struct, str):
        return {tree_struct}
    leaves = set()
    for child in tree_struct:
        leaves |= _get_all_leaves(child)
    return leaves

def _me_score(tree_struct, d_dict):
    """Compute Minimum Evolution score: sum of all pairwise distances implied by tree.
    For topology-only scoring, we use the sum of edge lengths estimated from the distance matrix.
    Here we approximate by summing d(i,j) for all pairs and comparing to tree-based estimate.
    Actually, for NNI with only topology, we can use the quartet-based score or 
    the sum of minimum spanning tree on the distance matrix restricted to the tree's splits.
    
    A simpler approach: use the total branch length estimated by least squares.
    But that's expensive. Instead, use the quartet score: for each quartet, check if the tree
    agrees with the shortest internal edge in the quartet distance matrix.
    
    For efficiency, let's use a proxy: the sum of all pairwise distances weighted by 
    whether the tree separates them or not. Actually, the standard ME score is:
    L = sum over all edges e of (len(e) * (n-1)) where len(e) is estimated.
    
    For topology comparison without branch lengths, a common proxy is the quartet score:
    Count how many quartets have the tree's topology matching the minimum-distance topology.
    """
    leaves = _get_all_leaves(tree_struct)
    leaf_list = sorted(leaves, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    n = len(leaf_list)
    
    if n < 4:
        return 0.0
    
    # Get splits from tree
    def get_splits(node, parent_set=None):
        """Yield all non-trivial splits as frozensets."""
        if isinstance(node, str):
            return
        if len(node) == 1:
            child = node[0]
            if isinstance(child, str):
                return
            else:
                yield from get_splits(child)
        else:
            # Internal node with 2 or 3 children
            if len(node) == 2:
                left_leaves = _get_all_leaves(node[0])
                right_leaves = _get_all_leaves(node[1])
                if 0 < len(left_leaves) < n:
                    comp = leaves - left_leaves
                    if min(left_leaves) < min(comp):
                        yield frozenset(left_leaves)
                    else:
                        yield frozenset(comp)
                yield from get_splits(node[0])
                yield from get_splits(node[1])
            elif len(node) == 3:
                for i in range(3):
                    other = [node[j] for j in range(3) if j != i]
                    other_leaves = _get_all_leaves(other[0]) | _get_all_leaves(other[1])
                    if 0 < len(other_leaves) < n:
                        comp = leaves - other_leaves
                        if min(other_leaves) < min(comp):
                            yield frozenset(other_leaves)
                        else:
                            yield frozenset(comp)
                    yield from get_splits(node[i])
    
    # Use quartet-based scoring for efficiency
    # For each quartet (a,b,c,d), the tree induces one of three topologies.
    # The ME-optimal topology is the one with the shortest internal edge, which corresponds
    # to the pairing with minimum d(a,b)+d(c,d), d(a,c)+d(b,d), or d(a,d)+d(b,c).
    
    score = 0.0
    count = 0
    
    # Sample quartets if too many
    if n > 15:
        import random
        random.seed(42)
        indices = list(range(n))
        num_quartets = min(1000, len(list(_itertools.combinations(range(n), 4))))
        quartets = random.sample(list(_itertools.combinations(range(n), 4)), min(num_quartets, len(list(_itertools.combinations(range(n), 4)))))
    else:
        import itertools as _itertools
        quartets = list(_itertools.combinations(range(n), 4))
    
    for (i, j, k, l) in quartets:
        a, b, c, d = leaf_list[i], leaf_list[j], leaf_list[k], leaf_list[l]
        
        # Determine which topology the tree induces for this quartet
        # Check if {a,b} is a split in the tree
        ab_split = frozenset({a, b})
        ac_split = frozenset({a, c})
        ad_split = frozenset({a, d})
        
        # Get all splits from tree
        tree_splits = set()
        for s in get_splits(tree_struct):
            tree_splits.add(s)
            tree_splits.add(leaves - s)
        
        # Determine the tree's quartet topology
        if ab_split in tree_splits or (leaves - ab_split) in tree_splits:
            tree_topo = (0, 1)  # a-b paired, c-d paired
        elif ac_split in tree_splits or (leaves - ac_split) in tree_splits:
            tree_topo = (0, 2)  # a-c paired, b-d paired
        else:
            tree_topo = (0, 3)  # a-d paired, b-c paired
        
        # Compute the three possible topologies' scores
        d_ab = d_dict.get((a, b), d_dict.get((b, a), 0))
        d_cd = d_dict.get((c, d), d_dict.get((d, c), 0))
        d_ac = d_dict.get((a, c), d_dict.get((c, a), 0))
        d_bd = d_dict.get((b, d), d_dict.get((d, b), 0))
        d_ad = d_dict.get((a, d), d_dict.get((d, a), 0))
        d_bc = d_dict.get((b, c), d_dict.get((c, b), 0))
        
        score_01 = d_ab + d_cd  # a-b, c-d
        score_02 = d_ac + d_bd  # a-c, b-d
        score_03 = d_ad + d_bc  # a-d, b-c
        
        min_score = min(score_01, score_02, score_03)
        
        if tree_topo == (0, 1):
            score += score_01
        elif tree_topo == (0, 2):
            score += score_02
        else:
            score += score_03
        
        count += 1
    
    return score / max(count, 1)

def _nni_refine(tree_struct, d_dict, labels, max_rounds=20):
    """NNI refinement using Minimum Evolution quartet scoring."""
    n = len(labels)
    if n < 4:
        return tree_struct
    
    current_tree = tree_struct
    current_score = _me_score(current_tree, d_dict)
    
    for round_num in range(max_rounds):
        if _time_left() < 3.0:
            break
        
        improved = False
        
        # Find all internal edges and try NNI moves
        # An NNI move is applied to an internal edge (u,v) where u and v are internal nodes
        # with 2 children each, and the edge connects them.
        # The three possible topologies for the quartet formed by the four subtrees.
        
        def find_nni_candidates(node, parent=None):
            """Yield (node, parent) pairs where NNI can be applied."""
            if isinstance(node, str):
                return
            if len(node) == 2:
                # Check if both children are internal (have 2 children)
                left, right = node
                left_is_internal = not isinstance(left, str) and len(left) == 2
                right_is_internal = not isinstance(right, str) and len(right) == 2
                
                if left_is_internal and right_is_internal:
                    yield (node, parent)
                
                yield from find_nni_candidates(left, node)
                yield from find_nni_candidates(right, node)
            elif len(node) == 3:
                for child in node:
                    yield from find_nni_candidates(child, node)
        
        # For each NNI candidate, try the two alternative topologies
        candidates = list(find_nni_candidates(current_tree))
        
        best_score = current_score
        best_tree = current_tree
        
        for (node, parent) in candidates:
            if _time_left() < 2.0:
                break
            
            # node has two children: left and right
            left, right = node[0], node[1]
            
            # left has two children: ll, lr
            # right has two children: rl, rr
            ll, lr = left[0], left[1]
            rl, rr = right[0], right[1]
            
            # Current topology: ((ll,lr),(rl,rr))
            # NNI alternatives:
            # Alt 1: ((ll,rl),(lr,rr))
            # Alt 2: ((ll,rr),(lr,rl))
            
            # Create alternative 1
            alt1_left = (ll, rl)
            alt1_right = (lr, rr)
            alt1_node = (alt1_left, alt1_right)
            
            # Create alternative 2
            alt2_left = (ll, rr)
            alt2_right = (lr, rl)
            alt2_node = (alt2_left, alt2_right)
            
            # Rebuild tree with alternative 1
            def replace_node(tree, target, replacement):
                """Replace target subtree with replacement in tree."""
                if tree == target:
                    return replacement
                if isinstance(tree, str):
                    return tree
                new_children = []
                for child in tree:
                    new_children.append(replace_node(child, target, replacement))
                return tuple(new_children)
            
            # We need to find the exact node object. Since we're using tuples,
            # we need a different approach. Let's rebuild from scratch.
            
            # Actually, let's just compute the score difference using quartets
            # that are affected by this NNI move. Only quartets that include
            # one leaf from each of the four subtrees (ll, lr, rl, rr) are affected.
            
            ll_leaves = _get_all_leaves(ll)
            lr_leaves = _get_all_leaves(lr)
            rl_leaves = _get_all_leaves(rl)
            rr_leaves = _get_all_leaves(rr)
            
            # For each combination of one leaf from each group, check if the
            # quartet topology changes
            score_diff = 0.0
            count = 0
            
            import random
            # Sample from each group if large
            ll_sample = list(ll_leaves)
            lr_sample = list(lr_leaves)
            rl_sample = list(rl_leaves)
            rr_sample = list(rr_leaves)
            
            # Limit sample size
            max_per_group = 5
            if len(ll_sample) > max_per_group:
                random.seed(round_num * 1000 + 1)
                ll_sample = random.sample(ll_sample, max_per_group)
            if len(lr_sample) > max_per_group:
                random.seed(round_num * 1000 + 2)
                lr_sample = random.sample(lr_sample, max_per_group)
            if len(rl_sample) > max_per_group:
                random.seed(round_num * 1000 + 3)
                rl_sample = random.sample(rl_sample, max_per_group)
            if len(rr_sample) > max_per_group:
                random.seed(round_num * 1000 + 4)
                rr_sample = random.sample(rr_sample, max_per_group)
            
            for a in ll_sample:
                for b in lr_sample:
                    for c in rl_sample:
                        for d in rr_sample:
                            # Current topology: (a,b) and (c,d) are paired
                            # Alt1: (a,c) and (b,d)
                            # Alt2: (a,d) and (b,c)
                            
                            d_ab = d_dict.get((a, b), d_dict.get((b, a), 0))
                            d_cd = d_dict.get((c, d), d_dict.get((d, c), 0))
                            d_ac = d_dict.get((a, c), d_dict.get((c, a), 0))
                            d_bd = d_dict.get((b, d), d_dict.get((d, b), 0))
                            d_ad = d_dict.get((a, d), d_dict.get((d, a), 0))
                            d_bc = d_dict.get((b, c), d_dict.get((c, b), 0))
                            
                            score_cur = d_ab + d_cd
                            score_alt1 = d_ac + d_bd
                            score_alt2 = d_ad + d_bc
                            
                            # Current tree has (a,b)|(c,d)
                            # If alt1 or alt2 has lower score, the move improves
                            best_alt = min(score_alt1, score_alt2)
                            score_diff += (best_alt - score_cur)
                            count += 1
            
            if count > 0:
                avg_diff = score_diff / count
                if avg_diff < -1e-9:
                    # This NNI move improves the score
                    # Rebuild the tree with this move
                    # We need to actually modify the tree structure
                    
                    # For now, just track that improvement exists
                    # Full implementation requires tree surgery
                    pass
        
        # For a complete implementation, we'd need to do tree surgery.
        # Given time constraints, let's return the NJ tree as is.
        # The quartet scoring can be used for a more sophisticated search.
        
        break
    
    return current_tree

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    d_dict, _ = _jc_distance_matrix(sequences)
    
    # Get NJ tree
    tree_struct = _nj_tree_struct(d_dict, labels)
    
    # Convert to Newick
    newick = _tree_to_newick(tree_struct) + ";"
    
    # NNI refinement (simplified - just returns NJ for now)
    # Full NNI would require tree surgery which is complex
    # The NJ tree is a solid baseline
    
    return newick
>>>>>>> REPLACE

Now let me implement a more complete NNI with actual tree surgery:

