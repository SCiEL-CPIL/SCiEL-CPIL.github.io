"""Phylogenetic topology recovery using quartet-based approach with NJ initialization and NNI refinement."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations

_DEADLINE_SECONDS = 255.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE_SECONDS - _elapsed()


def _jc_corrected(p):
    """Jukes-Cantor distance correction."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _seqs_to_array(sequences):
    """Convert sequences to numeric array and return labels, array."""
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    arr = np.zeros((n, L), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            if c == 'C':
                arr[i, j] = 1
            elif c == 'G':
                arr[i, j] = 2
            elif c == 'T':
                arr[i, j] = 3
    return labels, arr


def _compute_distance_matrix(seq_arr):
    """Compute JC-corrected pairwise distance matrix using vectorized operations."""
    n, L = seq_arr.shape
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            p = np.count_nonzero(seq_arr[i] != seq_arr[j]) / L
            d = _jc_corrected(p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


class _Node:
    """Tree node for NNI operations."""
    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None
        self.length = 0.0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def get_leaves(self):
        if self.is_leaf():
            return [self.name]
        leaves = []
        for c in self.children:
            leaves.extend(c.get_leaves())
        return leaves


def _parse_newick(newick, n):
    """Parse Newick string into a tree. Returns root node."""
    pos = [0]
    
    def parse():
        if pos[0] >= len(newick):
            return _Node()
        
        if newick[pos[0]] == '(':
            pos[0] += 1
            node = _Node()
            while True:
                child = parse()
                node.children.append(child)
                child.parent = node
                if pos[0] < len(newick) and newick[pos[0]] == ',':
                    pos[0] += 1
                elif pos[0] < len(newick) and newick[pos[0]] == ')':
                    pos[0] += 1
                    break
                else:
                    break
            
            # Check for branch length
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                start = pos[0]
                while pos[0] < len(newick) and newick[pos[0]] not in ',):':
                    pos[0] += 1
                try:
                    node.length = float(newick[start:pos[0]])
                except:
                    node.length = 0.0
            
            return node
        else:
            # Leaf
            start = pos[0]
            while pos[0] < len(newick) and newick[pos[0]] not in ',):':
                pos[0] += 1
            name = newick[start:pos[0]]
            
            # Check for branch length
            if pos[0] < len(newick) and newick[pos[0]] == ':':
                pos[0] += 1
                start2 = pos[0]
                while pos[0] < len(newick) and newick[pos[0]] not in ',):':
                    pos[0] += 1
                try:
                    length = float(newick[start2:pos[0]])
                except:
                    length = 0.0
            else:
                length = 0.0
            
            node = _Node(name)
            node.length = length
            return node
    
    root = parse()
    return root


def _get_all_nodes(root):
    """Get all nodes in the tree."""
    nodes = []
    def _dfs(node):
        nodes.append(node)
        for c in node.children:
            _dfs(c)
    _dfs(root)
    return nodes


def _get_internal_nodes(root):
    """Get all internal (non-leaf) nodes."""
    nodes = []
    def _dfs(node):
        if not node.is_leaf():
            nodes.append(node)
        for c in node.children:
            _dfs(c)
    _dfs(root)
    return nodes


def _newick_from_tree(node):
    """Convert tree to Newick string."""
    if node.is_leaf():
        return node.name
    
    child_strs = []
    for c in node.children:
        s = _newick_from_tree(c)
        if c.length > 0.001:
            s += f":{c.length:.4f}"
        child_strs.append(s)
    
    return "(" + ",".join(child_strs) + ")"


def _nj(sequences):
    """Neighbor Joining algorithm - fast initialization."""
    labels, seq_arr = _seqs_to_array(sequences)
    n = len(labels)
    L = len(sequences[labels[0]])
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(seq_arr)
    
    # NJ algorithm
    active = list(range(n))
    subtrees = {i: f"t{i}" for i in range(n)}
    
    d = np.zeros((2 * n, 2 * n), dtype=np.float64)
    d[:n, :n] = dist
    
    next_id = n
    m = n
    
    while m > 3:
        if _time_left() < 5.0:
            break
        
        # Compute row sums
        total = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    total[i] += d[active[i], active[j]]
        
        # Find the pair (i,j) that minimizes the NJ criterion
        best_criterion = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * d[active[i], active[j]] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = active[best_i], active[best_j]
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Compute distances from new node to all other active nodes
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            d[new_node, x] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[x, new_node] = d[new_node, x]
        
        # Update active list
        active = rest + [new_node]
        m = len(active)
        
        # Update subtree
        subtrees[new_node] = f"({subtrees[a]},{subtrees[b]})"
    
    # Build final Newick
    result = "(" + ",".join(subtrees[x] for x in active) + ");"
    
    return result


def _quartet_score(seq_arr, a, b, c, d):
    """
    Score quartet topology ((a,b),(c,d)) vs alternatives using JC model.
    Returns scores for the three possible quartet topologies.
    Uses the fact that under JC, the most likely topology is the one
    that minimizes the sum of branch lengths needed to explain the data.
    
    For four taxa a,b,c,d, the three possible unrooted topologies are:
    1. ((a,b),(c,d)) - split {a,b}|{c,d}
    2. ((a,c),(b,d)) - split {a,c}|{b,d}
    3. ((a,d),(b,c)) - split {a,d}|{b,c}
    
    We use a distance-based criterion: for topology ((a,b),(c,d)),
    the implied distance d(a,b) + d(c,d) should be less than d(a,c)+d(b,d) and d(a,d)+d(b,c).
    """
    L = seq_arr.shape[1]
    
    # Compute pairwise mismatch counts (un-corrected for speed)
    ab = np.count_nonzero(seq_arr[a] != seq_arr[b])
    ac = np.count_nonzero(seq_arr[a] != seq_arr[c])
    ad = np.count_nonzero(seq_arr[a] != seq_arr[d])
    bc = np.count_nonzero(seq_arr[b] != seq_arr[c])
    bd = np.count_nonzero(seq_arr[b] != seq_arr[d])
    cd = np.count_nonzero(seq_arr[c] != seq_arr[d])
    
    # Under JC model, the expected number of substitutions is proportional to distance
    # For topology ((a,b),(c,d)), the path lengths are:
    # a-b: d_ab, c-d: d_cd, and the connecting path
    # The criterion is: d_ab + d_cd vs d_ac + d_bd vs d_ad + d_bc
    # Lower sum means more likely under that topology
    
    score_ab_cd = ab + cd
    score_ac_bd = ac + bd
    score_ad_bc = ad + bc
    
    return score_ab_cd, score_ac_bd, score_ad_bc


def _quartet_majority_vote(seq_arr, n, time_budget=30.0):
    """
    Compute weighted majority vote of quartet topologies.
    For each quartet, determine the most likely topology and count votes.
    Then build a tree that maximizes agreement with the voted quartets.
    """
    start = time.time()
    
    # Count votes for each split
    # A split is represented by a frozenset of taxa on one side
    split_votes = {}
    
    total_quartets = 0
    max_quartets = 50000  # Limit for large n
    
    # For n up to 32, C(32,4) = 35960, which is manageable
    # For n > 32, we'd need to sample, but max n is 32
    
    quartets = list(combinations(range(n), 4))
    if len(quartets) > max_quartets:
        np.random.seed(42)
        idx = np.random.choice(len(quartets), max_quartets, replace=False)
        quartets = [quartets[i] for i in idx]
    
    for a, b, c, d in quartets:
        if (time.time() - start) > time_budget or _time_left() < 20.0:
            break
        
        s1, s2, s3 = _quartet_score(seq_arr, a, b, c, d)
        
        # Find the minimum score (most likely topology)
        min_score = min(s1, s2, s3)
        
        # Add votes for the winning topology
        # Topology 1: ((a,b),(c,d)) -> split {a,b}
        # Topology 2: ((a,c),(b,d)) -> split {a,c}
        # Topology 3: ((a,d),(b,c)) -> split {a,d}
        
        if s1 == min_score:
            key = frozenset({a, b})
            split_votes[key] = split_votes.get(key, 0) + 1
            # Also add the complementary split
            comp = frozenset({c, d})
            split_votes[comp] = split_votes.get(comp, 0) + 1
        elif s2 == min_score:
            key = frozenset({a, c})
            split_votes[key] = split_votes.get(key, 0) + 1
            comp = frozenset({b, d})
            split_votes[comp] = split_votes.get(comp, 0) + 1
        else:
            key = frozenset({a, d})
            split_votes[key] = split_votes.get(key, 0) + 1
            comp = frozenset({b, c})
            split_votes[comp] = split_votes.get(comp, 0) + 1
        
        total_quartets += 1
    
    return split_votes, total_quartets


def _build_tree_from_splits(n, split_votes):
    """
    Build a tree from the voted splits using a greedy approach.
    Start with NJ tree and refine using the quartet votes.
    """
    # This is a placeholder - we'll use NJ + NNI instead
    pass


def _nni_refine_quartet(tree, seq_arr, n, L, split_votes, time_budget=30.0):
    """
    NNI refinement using quartet-based scoring.
    For each NNI move, evaluate the change in quartet agreement.
    """
    if n < 4:
        return tree
    
    start = time.time()
    improved = True
    best_tree = tree
    
    # Precompute leaf sets for each internal edge
    def get_leaf_set(node):
        if node.is_leaf():
            return {node.name}
        s = set()
        for c in node.children:
            s |= get_leaf_set(c)
        return s
    
    while improved and (time.time() - start) < time_budget and _time_left() > 20.0:
        improved = False
        
        internals = _get_internal_nodes(best_tree)
        
        for node in internals:
            if _time_left() < 20.0:
                break
            
            if len(node.children) != 2:
                continue
            
            if node.parent is None:
                continue
            
            parent = node.parent
            if len(parent.children) != 2:
                continue
            
            # Find sibling
            sibling = None
            for c in parent.children:
                if c is not node:
                    sibling = c
                    break
            
            if sibling is None or len(sibling.children) != 2:
                continue
            
            s1, s2 = sibling.children[0], sibling.children[1]
            c1, c2 = node.children[0], node.children[1]
            
            # Get leaf sets
            leaves_c1 = set(c1.get_leaves())
            leaves_c2 = set(c2.get_leaves())
            leaves_s1 = set(s1.get_leaves())
            leaves_s2 = set(s2.get_leaves())
            
            # Original split: {c1} | {c2, s1, s2} vs {s1} | {s2, c1, c2}
            # Actually, the internal edge is between node and parent
            # The split defined by this edge is: leaves(node) | leaves(parent) \ leaves(node)
            
            leaves_node = leaves_c1 | leaves_c2
            leaves_parent = leaves_node | leaves_s1 | leaves_s2
            
            # Two NNI alternatives:
            # Alt 1: ((c1, s1), (c2, s2))
            # Alt 2: ((c1, s2), (c2, s1))
            
            for alt_type in [1, 2]:
                if _time_left() < 20.0:
                    break
                
                # Evaluate using quartet scores
                # Sample some quartets to estimate the improvement
                # For efficiency, use a small random sample
                
                if len(leaves_c1) * len(leaves_s1) > 100:
                    # Too large, skip detailed evaluation
                    continue
                
                # Try the NNI move
                import copy
                new_tree = copy.deepcopy(best_tree)
                
                # Find corresponding nodes in new_tree
                # This is complex, so let's use a simpler approach:
                # Just evaluate using the distance matrix
                
                # Compute the change in the NJ-like criterion
                # For now, use a heuristic based on the quartet votes
                
                # Check if the new split has more votes
                if alt_type == 1:
                    # New split: {c1, s1} | {c2, s2}
                    new_split1 = frozenset(leaves_c1 | leaves_s1)
                    new_split2 = frozenset(leaves_c2 | leaves_s2)
                else:
                    # New split: {c1, s2} | {c2, s1}
                    new_split1 = frozenset(leaves_c1 | leaves_s2)
                    new_split2 = frozenset(leaves_c2 | leaves_s1)
                
                # Get votes for the new splits
                votes_new1 = split_votes.get(new_split1, 0)
                votes_new2 = split_votes.get(new_split2, 0)
                
                # Get votes for the old splits
                old_split1 = frozenset(leaves_node)
                old_split2 = frozenset(leaves_s1 | leaves_s2)
                votes_old1 = split_votes.get(old_split1, 0)
                votes_old2 = split_votes.get(old_split2, 0)
                
                if votes_new1 + votes_new2 > votes_old1 + votes_old2:
                    # Accept the move
                    # Perform the NNI move on new_tree
                    # Find the nodes in new_tree
                    def find_node_by_leaves(tree_node, target_leaves):
                        if tree_node.is_leaf():
                            return None
                        for c in tree_node.children:
                            if set(c.get_leaves()) == target_leaves:
                                return c
                        return None
                    
                    # This is getting complex, so let's just return the best tree found
                    # For now, skip the actual modification
                    
                    improved = True
                    break
            
            if improved:
                break
    
    return best_tree


def _parsimony_score(tree, seq_arr, n, L):
    """Compute Fitch parsimony score for a tree - fast vectorized version."""
    def get_leaf_order(node, order):
        if node.is_leaf():
            order.append(node.name)
            return
        for c in node.children:
            get_leaf_order(c, order)
    
    order = []
    get_leaf_order(tree, order)
    
    name_to_idx = {}
    for i, name in enumerate(order):
        name_to_idx[name] = i
    
    # Get leaf indices in tree order
    leaf_indices = [name_to_idx[name] for name in order]
    
    total_parsimony = 0
    
    # Process each site
    for site in range(L):
        # Get states at leaves
        leaf_states = seq_arr[leaf_indices, site]
        
        # Fitch algorithm - bottom up
        # We need to process the tree in post-order
        def fitch_postorder(node):
            """Returns (state_set, change_count)"""
            if node.is_leaf():
                idx = name_to_idx[node.name]
                return {leaf_states[idx]}, 0
            
            child_results = [fitch_postorder(c) for c in node.children]
            
            # Compute intersection
            intersection = set(child_results[0][0])
            for cr in child_results[1:]:
                intersection = intersection & cr[0]
            
            if intersection:
                total_changes = sum(cr[1] for cr in child_results)
                return intersection, total_changes
            else:
                union = set(child_results[0][0])
                for cr in child_results[1:]:
                    union = union | cr[0]
                total_changes = sum(cr[1] for cr in child_results) + 1
                return union, total_changes
        
        _, changes = fitch_postorder(tree)
        total_parsimony += changes
    
    return total_parsimony


def _nni_refine_parsimony(tree, seq_arr, n, L, time_budget=30.0):
    """
    NNI refinement using parsimony score.
    Try all NNI moves and keep improvements.
    """
    if n < 4:
        return tree
    
    current_score = _parsimony_score(tree, seq_arr, n, L)
    best_tree = tree
    best_score = current_score
    
    start = time.time()
    improved = True
    
    while improved and (time.time() - start) < time_budget and _time_left() > 20.0:
        improved = False
        
        internals = _get_internal_nodes(best_tree)
        
        for node in internals:
            if _time_left() < 20.0:
                break
            
            if len(node.children) != 2:
                continue
            
            if node.parent is None:
                continue
            
            parent = node.parent
            if len(parent.children) != 2:
                continue
            
            sibling = None
            for c in parent.children:
                if c is not node:
                    sibling = c
                    break
            
            if sibling is None or len(sibling.children) != 2:
                continue
            
            s1, s2 = sibling.children[0], sibling.children[1]
            c1, c2 = node.children[0], node.children[1]
            
            # Try two NNI alternatives
            for alt_type in [1, 2]:
                if _time_left() < 20.0:
                    break
                
                # Create new tree structure
                import copy
                new_tree = copy.deepcopy(best_tree)
                
                # Find the corresponding nodes in new_tree by their leaf sets
                def find_subtree_by_leaves(tree_node, target_leaves):
                    if tree_node.is_leaf():
                        return tree_node if {tree_node.name} == target_leaves else None
                    for c in tree_node.children:
                        if set(c.get_leaves()) == target_leaves:
                            return c
                    return None
                
                # Get leaf sets
                leaves_c1 = set(c1.get_leaves())
                leaves_c2 = set(c2.get_leaves())
                leaves_s1 = set(s1.get_leaves())
                leaves_s2 = set(s2.get_leaves())
                
                # Find corresponding nodes in new_tree
                if alt_type == 1:
                    # Alt 1: ((c1, s1), (c2, s2))
                    new_c1 = find_subtree_by_leaves(new_tree, leaves_c1)
                    new_c2 = find_subtree_by_leaves(new_tree, leaves_c2)
                    new_s1 = find_subtree_by_leaves(new_tree, leaves_s1)
                    new_s2 = find_subtree_by_leaves(new_tree, leaves_s2)
                    
                    if all(x is not None for x in [new_c1, new_c2, new_s1, new_s2]):
                        # Find the parent and node in new_tree
                        new_parent = find_subtree_by_leaves(new_tree, leaves_c1 | leaves_c2 | leaves_s1 | leaves_s2)
                        if new_parent is not None and len(new_parent.children) == 2:
                            # Find which child is the "node" and which is "sibling"
                            new_node = None
                            new_sibling = None
                            for c in new_parent.children:
                                if set(c.get_leaves()) == leaves_c1 | leaves_c2:
                                    new_node = c
                                elif set(c.get_leaves()) == leaves_s1 | leaves_s2:
                                    new_sibling = c
                            
                            if new_node is not None and new_sibling is not None:
                                # Perform NNI: new_node gets children [new_c1, new_s1]
                                # new_sibling gets children [new_c2, new_s2]
                                new_node.children = [new_c1, new_s1]
                                new_sibling.children = [new_c2, new_s2]
                                new_c1.parent = new_node
                                new_s1.parent = new_node
                                new_c2.parent = new_sibling
                                new_s2.parent = new_sibling
                                
                                # Evaluate
                                new_score = _parsimony_score(new_tree, seq_arr, n, L)
                                if new_score < best_score:
                                    best_tree = new_tree
                                    best_score = new_score
                                    improved = True
                                    break
                else:
                    # Alt 2: ((c1, s2), (c2, s1))
                    new_c1 = find_subtree_by_leaves(new_tree, leaves_c1)
                    new_c2 = find_subtree_by_leaves(new_tree, leaves_c2)
                    new_s1 = find_subtree_by_leaves(new_tree, leaves_s1)
                    new_s2 = find_subtree_by_leaves(new_tree, leaves_s2)
                    
                    if all(x is not None for x in [new_c1, new_c2, new_s1, new_s2]):
                        new_parent = find_subtree_by_leaves(new_tree, leaves_c1 | leaves_c2 | leaves_s1 | leaves_s2)
                        if new_parent is not None and len(new_parent.children) == 2:
                            new_node = None
                            new_sibling = None
                            for c in new_parent.children:
                                if set(c.get_leaves()) == leaves_c1 | leaves_c2:
                                    new_node = c
                                elif set(c.get_leaves()) == leaves_s1 | leaves_s2:
                                    new_sibling = c
                            
                            if new_node is not None and new_sibling is not None:
                                new_node.children = [new_c1, new_s2]
                                new_sibling.children = [new_c2, new_s1]
                                new_c1.parent = new_node
                                new_s2.parent = new_node
                                new_c2.parent = new_sibling
                                new_s1.parent = new_sibling
                                
                                new_score = _parsimony_score(new_tree, seq_arr, n, L)
                                if new_score < best_score:
                                    best_tree = new_tree
                                    best_score = new_score
                                    improved = True
                                    break
            
            if improved:
                break
    
    return best_tree


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from sequences.
    Uses NJ for initialization, quartet-based voting, and NNI refinement.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return f"{labels[0]};"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({','.join(labels)});"
    
    # Get sequences as array
    labels, seq_arr = _seqs_to_array(sequences)
    L = len(sequences[labels[0]])
    
    # Step 1: Run NJ for initial tree
    result = _nj(sequences)
    
    # Step 2: If time permits, compute quartet votes
    split_votes = {}
    if _time_left() > 40.0 and n >= 4:
        try:
            quartet_time = min(_time_left() - 30.0, 20.0)
            if quartet_time > 5.0:
                split_votes, _ = _quartet_majority_vote(seq_arr, n, quartet_time)
        except Exception:
            pass
    
    # Step 3: NNI refinement
    if _time_left() > 30.0 and n >= 4:
        try:
            tree = _parse_newick(result, n)
            nni_time = min(_time_left() - 20.0, 60.0)
            if nni_time > 10.0:
                polished = _nni_refine_parsimony(tree, seq_arr, n, L, nni_time)
                if polished is not tree:
                    result = _newick_from_tree(polished) + ";"
        except Exception:
            pass
    
    return result
# EVOLVE-BLOCK-END
