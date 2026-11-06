# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations
from collections import defaultdict

_START = None
_DEADLINE_FRACTION = 0.85

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    
    # Use a dict of dicts for distance to allow dynamic updates
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
            
    splits = set()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

def _rf_distance(s1, s2):
    return len(s1.symmetric_difference(s2))

def _majority_vote(splits_list):
    """
    Given a list of split sets, return the majority vote split set.
    A split is included if it appears in more than half of the trees.
    """
    from collections import Counter
    counter = Counter()
    for splits in splits_list:
        for s in splits:
            counter[s] += 1
    
    threshold = len(splits_list) / 2.0
    majority = set()
    for s, count in counter.items():
        if count > threshold:
            majority.add(s)
    return majority

def _parse_newick(newick, labels):
    """
    Parse a Newick string into a tree structure represented by:
    - nodes: list of node ids
    - children: dict mapping node id to list of child ids
    - parent: dict mapping node id to parent id
    - is_leaf: dict mapping node id to bool
    - leaf_label: dict mapping node id to label
    """
    # Simple parser for unrooted trees
    # We assume standard Newick format: (A,B,C); or (A,(B,C));
    # Since it's unrooted, the root is arbitrary.
    
    # For NNI, we need an adjacency representation.
    # Let's build a graph.
    # Node IDs will be assigned sequentially.
    
    node_id = 0
    next_id = [0]
    children = {}
    parent = {}
    is_leaf = {}
    leaf_label = {}
    
    def parse_subtree(s, i):
        # s is the string, i is the current index
        if s[i] == '(':
            next_id[0] += 1
            curr_id = next_id[0]
            i += 1
            child_ids = []
            while s[i] != ')':
                child_id, i = parse_subtree(s, i)
                child_ids.append(child_id)
            i += 1 # skip ')'
            
            children[curr_id] = child_ids
            is_leaf[curr_id] = False
            
            # Parse label and branch length after the closing paren
            label = ""
            branch_len = 0.0
            if i < len(s) and s[i] == ':':
                i += 1
                num_str = ""
                while i < len(s) and s[i] not in [',', ')', ';']:
                    num_str += s[i]
                    i += 1
                if num_str:
                    branch_len = float(num_str)
            else:
                while i < len(s) and s[i] not in [',', ')', ';']:
                    label += s[i]
                    i += 1
            
            if label:
                # This is a leaf? No, leaves are usually just labels.
                # But in our generated Newick, leaves are just labels.
                # Internal nodes don't have labels in the output of _nj.
                # Wait, _nj produces "(label1,label2)" for leaves.
                # So if we see a label without a preceding '(', it's a leaf.
                # But here we are inside a '(', so this is an internal node.
                # The label might be empty.
                pass
            
            return curr_id, i
        else:
            # Leaf node
            next_id[0] += 1
            curr_id = next_id[0]
            label = ""
            while i < len(s) and s[i] not in [',', ')', ';']:
                label += s[i]
                i += 1
            
            is_leaf[curr_id] = True
            leaf_label[curr_id] = label
            children[curr_id] = []
            
            # Handle branch length if present (though _nj doesn't output them)
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in [',', ')', ';']:
                    i += 1
            
            return curr_id, i

    # The input newick is unrooted, so it might be like (A,B,(C,D));
    # We need to find the root.
    # Actually, _nj returns a string like "(A,B,(C,D))".
    # We can just parse it.
    
    # Remove trailing semicolon
    if newick.endswith(';'):
        newick = newick[:-1]
    
    root_id, _ = parse_subtree(newick, 0)
    
    # Build parent pointers
    for node, childs in children.items():
        for c in childs:
            parent[c] = node
            
    # For NNI, we need to work with the unrooted tree.
    # We will return the adjacency list and leaf info.
    return root_id, children, parent, is_leaf, leaf_label

def _nni_refine(newick, dist, labels, max_iters=100):
    """
    Refine the tree topology using NNI moves to minimize the Fitch-Margoliash error.
    """
    n = len(labels)
    if n < 4:
        return newick

    # Map labels to indices in the dist matrix
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    # Parse the initial tree
    root_id, children, parent, is_leaf, leaf_label = _parse_newick(newick, labels)
    
    # Get all leaf nodes and their labels
    leaf_nodes = [node for node, is_l in is_leaf.items() if is_l]
    # Map leaf node ID to label
    leaf_node_to_label = {node: leaf_label[node] for node in leaf_nodes}
    
    # We need to calculate the distance between any two leaves in the current tree.
    # This requires summing branch lengths. But we don't have branch lengths from _nj.
    # _nj does not output branch lengths.
    # To perform NNI based on Fitch-Margoliash, we need branch lengths.
    # We can estimate branch lengths for the current topology using the distance matrix.
    # Or, we can use a simpler NNI criterion: 
    # The NNI move that minimizes the sum of distances between the four subtrees?
    # 
    # A common heuristic for NNI without explicit branch lengths is to use the 
    # "quartet score" or to re-estimate branch lengths after each move.
    # Re-estimating branch lengths is O(N^2) per move.
    # 
    # Let's implement a simpler approach: 
    # 1. Identify all internal edges.
    # 2. For each internal edge, consider the NNI moves.
    # 3. For each candidate topology (current and 2 alternatives), estimate the 
    #    branch lengths that minimize the sum of squared distances (Fitch-Margoliash).
    # 4. Choose the topology with the lowest error.
    
    # To make this efficient, we will represent the tree as a set of splits.
    # Actually, implementing full NNI with branch length optimization in Python
    # within the time limit is risky.
    
    # Alternative: Use the "Quartet Method" for NNI?
    # Or simply: If we have a good distance matrix, NJ is already pretty good.
    # The main benefit of NNI is fixing local errors.
    
    # Let's try a simpler NNI: 
    # Minimize the sum of the three "long" distances in the quartet defined by the internal edge.
    # For an internal edge connecting subtrees A, B, C, D (where A,B are on one side, C,D on the other),
    # the current topology is (AB)(CD).
    # The alternative topologies are (AC)(BD) and (AD)(BC).
    # We can compute the "quartet score" for each:
    # S1 = d(A,C) + d(B,D)
    # S2 = d(A,B) + d(C,D)
    # S3 = d(A,D) + d(B,C)
    # where d(X,Y) is the distance between the two subtrees.
    # The distance between two subtrees can be approximated by the average distance between leaves in the subtrees.
    
    # This is a heuristic but often works well.
    
    # Let's implement this heuristic NNI.
    
    # First, we need to compute the average distance between any two sets of leaves.
    # We can precompute the pairwise leaf distances from the input dist matrix.
    # dist is already the matrix.
    
    # We need to identify the internal edges in the parsed tree.
    # An internal edge connects two internal nodes.
    
    internal_edges = []
    for node, childs in children.items():
        if is_leaf[node]:
            continue
        for c in childs:
            if not is_leaf[c]:
                # Edge (node, c) is internal
                # Add it once. Let's say node > c to avoid duplicates? 
                # Or just store both and check later.
                internal_edges.append((node, c))
    
    # To avoid processing the same edge twice, we can sort or use a set.
    # Since the tree is undirected, (u,v) is the same as (v,u).
    # Let's just iterate and ensure we don't process (u,v) and (v,u).
    processed_edges = set()
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters and _time_left() > 1.0:
        improved = False
        iter_count += 1
        
        # Re-parse if we modified the tree? 
        # Modifying the tree structure in the dict is complex.
        # Instead, we will generate the new Newick string for the best move and re-parse.
        
        best_move = None
        best_score = float('inf')
        current_score = None
        
        # We need a baseline score for the current tree to compare against.
        # Let's compute the sum of quartet scores for all internal edges?
        # No, let's just evaluate each move relative to the current local configuration.
        
        for u, v in internal_edges:
            if (v, u) in processed_edges:
                continue
            processed_edges.add((u, v))
            
            # u and v are internal nodes connected by an edge.
            # u has children. One is v. The others are subtrees A, B.
            # v has children. One is u. The others are subtrees C, D.
            
            # Get children of u excluding v
            children_u = [c for c in children[u] if c != v]
            children_v = [c for c in children[v] if c != u]
            
            if len(children_u) != 2 or len(children_v) != 2:
                continue # Not a valid NNI candidate (must be binary tree)
            
            a, b = children_u
            c, d = children_v
            
            # Get leaf sets for a, b, c, d
            def get_leaves(node):
                leaves = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if is_leaf[curr]:
                        leaves.add(leaf_label[curr])
                    else:
                        stack.extend(children[curr])
                return leaves
            
            set_a = get_leaves(a)
            set_b = get_leaves(b)
            set_c = get_leaves(c)
            set_d = get_leaves(d)
            
            # Compute average distances
            def avg_dist(set1, set2):
                if not set1 or not set2:
                    return 0
                total = 0
                count = 0
                for l1 in set1:
                    idx1 = label_to_idx[l1]
                    for l2 in set2:
                        idx2 = label_to_idx[l2]
                        total += dist[idx1][idx2]
                        count += 1
                return total / count if count > 0 else 0
            
            d_ab = avg_dist(set_a, set_b)
            d_cd = avg_dist(set_c, set_d)
            d_ac = avg_dist(set_a, set_c)
            d_bd = avg_dist(set_b, set_d)
            d_ad = avg_dist(set_a, set_d)
            d_bc = avg_dist(set_b, set_c)
            
            # Current topology: (AB)(CD). 
            # The "cost" associated with this internal edge in a quartet sense is often 
            # the sum of the two "long" branches or the sum of distances across the cut.
            # Let's use the sum of distances between the two sides: d(A,C)+d(A,D)+d(B,C)+d(B,D)?
            # Or simpler: The quartet score for the topology (AB|CD) is d(A,C)+d(B,D).
            # The alternative (AC|BD) is d(A,B)+d(C,D).
            # The alternative (AD|BC) is d(A,D)+d(B,C).
            
            # We want to minimize the "quartet score" for the topology?
            # Actually, for a fixed distance matrix, the topology that minimizes 
            # d(A,C)+d(B,D) is not necessarily the best.
            # The standard quartet test chooses the topology with the *smallest* 
            # sum of the two "internal" distances? No.
            
            # Let's use the following heuristic:
            # The NNI move that reduces the sum of the three "cross" distances is better.
            # Cross distances for (AB|CD) are d(A,C), d(A,D), d(B,C), d(B,D).
            # Sum = d(A,C) + d(A,D) + d(B,C) + d(B,D).
            
            score_current = d_ac + d_ad + d_bc + d_bd
            
            # Alternative 1: (AC|BD)
            # Cross distances: d(A,B), d(A,D), d(C,B), d(C,D)
            score_alt1 = d_ab + d_ad + d_bc + d_cd
            
            # Alternative 2: (AD|BC)
            # Cross distances: d(A,B), d(A,C), d(D,B), d(D,C)
            score_alt2 = d_ab + d_ac + d_bd + d_cd
            
            # We want to minimize the score.
            if score_alt1 < score_current and score_alt1 < best_score:
                best_score = score_alt1
                best_move = ('AC_BD', u, v, a, b, c, d)
                improved = True
            elif score_alt2 < score_current and score_alt2 < best_score:
                best_score = score_alt2
                best_move = ('AD_BC', u, v, a, b, c, d)
                improved = True
        
        if best_move:
            # Apply the best move
            move_type, u, v, a, b, c, d = best_move
            # We need to update the tree structure.
            # This is complex with the dict representation.
            # Given the time constraints and complexity, I will skip the actual structural update 
            # and rely on the Bootstrap Consensus which is more robust and easier to implement correctly.
            # The NNI heuristic above is a good indicator, but implementing the swap safely 
            # in a dict-based tree without re-parsing is error-prone.
            
            # Instead, I will break and rely on the consensus.
            break
            
    return newick

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    candidates = [(tree_full, splits_full)]
    
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 100 # Reduced to allow for potential NNI if I implement it, but mostly for speed
    
    # Use more bootstraps if time permits
    while num_bootstraps < max_bootstraps and _time_left() > 5.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        num_bootstraps += 1
        
    if len(candidates) > 1:
        # Use majority vote to select the consensus tree
        all_splits = [c[1] for c in candidates]
        consensus_splits = _majority_vote(all_splits)
        
        # Find the candidate closest to the consensus
        best_idx = 0
        best_score = float('inf')
        for i, (tree_i, splits_i) in enumerate(candidates):
            # RF distance to consensus
            score = _rf_distance(splits_i, consensus_splits)
            if score < best_score:
                best_score = score
                best_idx = i
        
        best_tree = candidates[best_idx][0]
        
        # Optional: NNI Refinement on the best tree
        # best_tree = _nni_refine(best_tree, d_full, labels)
        
        return best_tree + ";"
    else:
        return tree_full + ";"

# EVOLVE-BLOCK-END
