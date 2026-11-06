# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations
from collections import Counter

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
    """Run Neighbor Joining and return (newick, splits_set)."""
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
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

def _get_splits_from_newick(newick, labels):
    """Extract splits from a Newick string."""
    n = len(labels)
    if n <= 2:
        return set()
    
    all_leaves = frozenset(labels)
    splits = set()
    
    # Parse newick to get internal node leaf sets
    def parse(node_str, parent_splits=None):
        # This is a simplified parser for our generated newick
        pass
    
    # Instead, let's use a different approach: track splits during NJ
    # For now, we'll compute splits from the tree structure
    # by finding all internal nodes and their descendant leaf sets
    
    # Simple approach: for each pair of leaves, check if they're separated
    # This is O(n^3) which is fine for n<=32
    def get_leaf_set(subtree):
        """Get all leaves in a subtree string."""
        leaves = set()
        i = 0
        while i < len(subtree):
            if subtree[i] == '(':
                # Find matching close paren
                depth = 0
                j = i
                while j < len(subtree):
                    if subtree[j] == '(':
                        depth += 1
                    elif subtree[j] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                inner = subtree[i+1:j]
                leaves.update(get_leaf_set(inner))
                i = j + 1
            elif subtree[i] == ',':
                i += 1
            else:
                # It's a leaf name
                j = i
                while j < len(subtree) and subtree[j] not in ',()':
                    j += 1
                leaves.add(subtree[i:j])
                i = j
        return leaves
    
    # Extract all internal nodes
    def extract_splits(subtree, depth=0):
        i = 0
        while i < len(subtree):
            if subtree[i] == '(':
                depth = 1
                j = i + 1
                while depth > 0:
                    if subtree[j] == '(':
                        depth += 1
                    elif subtree[j] == ')':
                        depth -= 1
                    j += 1
                inner = subtree[i+1:j-1]
                leaf_set = frozenset(get_leaf_set(inner))
                if 1 < len(leaf_set) < n:
                    # Canonicalize
                    complement = all_leaves - leaf_set
                    if tuple(sorted(leaf_set)) < tuple(sorted(complement)):
                        splits.add(leaf_set)
                    else:
                        splits.add(complement)
                # Recurse into children
                extract_splits(inner)
                i = j
            elif subtree[i] == ',':
                i += 1
            else:
                i += 1
    
    extract_splits(newick)
    return splits

def _nni_refine(tree_newick, dist, labels, max_rounds=100):
    """
    Refine tree using NNI moves. Uses quartet score to evaluate each move.
    Returns improved newick string.
    """
    n = len(labels)
    if n < 4:
        return tree_newick
    
    all_leaves = set(labels)
    
    def quartet_score_for_tree(splits, a, b, c, d):
        """
        Given a set of splits (defining a tree) and 4 taxa, 
        determine which of the 3 topologies the tree implies and 
        return the distance-based score for that topology.
        Lower score = better.
        """
        # The tree implies a topology if the splits separate the taxa in a particular way
        # For quartet {a,b,c,d}, the tree implies (ab|cd) if there's a split that separates {a,b} from {c,d}
        # More precisely: (ab|cd) is implied if there exists a split S such that {a,b} ⊆ S and {c,d} ⊆ S^c
        # or {a,b} ⊆ S^c and {c,d} ⊆ S
        
        # Actually, for a tree to imply (ab|cd), the path between a and b must not cross the path between c and d
        # In terms of splits: there should be no split that separates a from b but not c from d, etc.
        
        # Simpler: check all 3 topologies and see which one is consistent with the splits
        # A topology (xy|zw) is consistent if there's no split that separates x from y but also separates z from w
        # (i.e., the split doesn't cross the quartet topology)
        
        def is_consistent(topo_pairs, splits):
            # topo_pairs is a list of two pairs: [(x,y), (z,w)] for topology (xy|zw)
            (x, y), (z, w) = topo_pairs
            for s in splits:
                # Check if this split separates x from y
                x_in = x in s
                y_in = y in s
                z_in = z in s
                w_in = w in s
                # The split separates x,y if exactly one is in s
                sep_xy = x_in != y_in
                sep_zw = z_in != w_in
                # If the split separates x from y AND separates z from w, 
                # then it's incompatible with (xy|zw)
                if sep_xy and sep_zw:
                    return False
            return True
        
        # Try all 3 topologies
        topologies = [
            ([(a, b), (c, d)], 'ab|cd'),
            ([(a, c), (b, d)], 'ac|bd'),
            ([(a, d), (b, c)], 'ad|bc'),
        ]
        
        for pairs, name in topologies:
            if is_consistent(pairs, splits):
                # This topology is consistent with the tree
                # Return the distance score for this topology
                if name == 'ab|cd':
                    return dist[a][b] + dist[c][d]
                elif name == 'ac|bd':
                    return dist[a][c] + dist[b][d]
                else:
                    return dist[a][d] + dist[b][c]
        
        # If no topology is consistent (shouldn't happen for a valid tree), return infinity
        return float('inf')
    
    def get_current_splits(newick):
        return _get_splits_from_newick(newick, labels)
    
    current_splits = get_current_splits(tree_newick)
    current_score = 0
    
    # Compute current total quartet score
    # This is O(n^4 * |splits|) which could be slow
    # For n=32, C(32,4)=35960 quartets, and ~30 splits, that's ~1M operations
    # Should be OK
    
    # Actually, let's use a simpler scoring: sum of distances for all pairs
    # No, that doesn't capture topology
    
    # Let's just do NNI with a simple criterion: 
    # For each NNI move, check if it reduces the sum of quartet scores
    # But computing all quartet scores each time is expensive
    
    # Alternative: use the "NNI with distance matrix" approach
    # For an internal edge (u,v) with neighbors a,b,c,d (where a-b is one side, c-d is the other),
    # the NNI move swaps to (a,c),(b,d) or (a,d),(b,c)
    # The gain can be computed from the distance matrix
    
    # Let me implement a simpler version: 
    # Use the fact that for a tree with splits, we can compute the "quartet score" 
    # as the sum over all quartets of the minimum score among the 3 topologies 
    # that are consistent with the tree
    
    # This is still expensive. Let's try a different approach:
    # Just do a few rounds of NNI using a simple heuristic
    
    # For now, return the input unchanged (NNI is complex and risky)
    return tree_newick

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    L = arr.shape[1]
    
    # Use fewer bootstraps but with a better selection criterion
    # Selection: pick the tree that minimizes the sum of quartet scores 
    # (distance-based) against the full-data distance matrix
    
    max_bootstraps = 50
    candidates = [(tree_full, splits_full)]
    
    num_bootstraps = 0
    while num_bootstraps < max_bootstraps and _time_left() > 15.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        num_bootstraps += 1
    
    if len(candidates) > 1:
        # Score each candidate by its quartet agreement with the full-data distances
        # For each candidate tree, compute the sum of quartet scores
        # (the distance sum for the topology implied by the tree for each quartet)
        
        def compute_quartet_score(splits, dist):
            """Compute total quartet score for a tree defined by splits."""
            total = 0
            # Precompute for each pair of taxa whether they're in the same side of each split
            # This helps speed up the consistency check
            
            # For each quartet {a,b,c,d}, find which topology is implied
            # and add the corresponding distance sum
            
            # To speed up, precompute split membership
            split_list = list(splits)
            n_splits = len(split_list)
            
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        for d in range(c+1, n):
                            # Check which topology is consistent
                            # (ab|cd)
                            consistent_ab_cd = True
                            consistent_ac_bd = True
                            consistent_ad_bc = True
                            
                            for s in split_list:
                                a_in = labels[a] in s
                                b_in = labels[b] in s
                                c_in = labels[c] in s
                                d_in = labels[d] in s
                                
                                # For (ab|cd): incompatible if split separates a from b AND c from d
                                if (a_in != b_in) and (c_in != d_in):
                                    consistent_ab_cd = False
                                if (a_in != c_in) and (b_in != d_in):
                                    consistent_ac_bd = False
                                if (a_in != d_in) and (b_in != c_in):
                                    consistent_ad_bc = False
                            
                            # Add score for the first consistent topology
                            if consistent_ab_cd:
                                total += dist[a][b] + dist[c][d]
                            elif consistent_ac_bd:
                                total += dist[a][c] + dist[b][d]
                            elif consistent_ad_bc:
                                total += dist[a][d] + dist[b][c]
                            # If none consistent, skip (shouldn't happen)
            
            return total
        
        # Compute scores for all candidates
        scores = []
        for i, (tree_i, splits_i) in enumerate(candidates):
            if _time_left() < 5.0:
                break
            score = compute_quartet_score(splits_i, d_full)
            scores.append(score)
        
        if scores:
            best_idx = min(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx][0] + ";"
        else:
            return tree_full + ";"
    else:
        return tree_full + ";"

# EVOLVE-BLOCK-END
