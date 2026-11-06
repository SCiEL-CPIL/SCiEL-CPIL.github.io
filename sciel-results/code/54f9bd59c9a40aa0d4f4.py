# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = None
_DEADLINE_FRACTION = 0.85
_TOTAL_BUDGET = 270.0

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TOTAL_BUDGET * _DEADLINE_FRACTION - (time.time() - _START))

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

def _get_splits_from_tree(tree_str, all_labels):
    """Extract splits from a Newick tree string."""
    labels_set = set(all_labels)
    n = len(labels_set)
    
    def parse_tree(s, pos=0):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse_tree(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                else:
                    break
            pos += 1
            return ('node', children), pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',;':
                end += 1
            label = s[pos:end].strip()
            return ('leaf', label), pos
    
    node, _ = parse_tree(tree_str)
    
    def get_leaf_set(n):
        if n[0] == 'leaf':
            return frozenset([n[1]])
        else:
            result = frozenset()
            for child in n[1]:
                result = result | get_leaf_set(child)
            return result
    
    splits = set()
    def traverse(n):
        if n[0] == 'node':
            leaf_set = get_leaf_set(n)
            if 1 < len(leaf_set) < n:
                side1 = leaf_set
                side2 = frozenset(labels_set - leaf_set)
                if tuple(sorted(side1)) < tuple(sorted(side2)):
                    splits.add(side1)
                else:
                    splits.add(side2)
            for child in n[1]:
                traverse(child)
    
    traverse(node)
    return splits

def _build_tree_from_splits(splits, labels):
    """Build a Newick tree from a set of splits using recursive partitioning."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    label_set = frozenset(labels)
    n = len(labels)
    
    best_split = None
    best_balance = 0
    
    for s in splits:
        side1 = s & label_set
        side2 = label_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            balance = min(len(side1), len(side2))
            if balance > best_balance:
                best_balance = balance
                best_split = side1
    
    if best_split is None:
        return "(" + ",".join(labels) + ")"
    
    side1 = sorted(best_split, key=lambda x: int(x[1:]))
    side2 = sorted(label_set - best_split, key=lambda x: int(x[1:]))
    
    tree1 = _build_tree_from_splits(splits, side1)
    tree2 = _build_tree_from_splits(splits, side2)
    
    return f"({tree1},{tree2})"

def _dpp_score_vectorized(splits, arr, n, L):
    """
    Compute DPP score using vectorized operations.
    Lower is better.
    """
    if not splits or n < 3:
        return 0.0
    
    # Precompute pairwise difference matrix
    # diff[i,j] = number of sites where taxa i and j differ
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    total_score = 0.0
    for split in splits:
        side1 = np.array([1 if l in split else 0 for l in range(n)])
        n1 = side1.sum()
        n2 = n - n1
        if n1 == 0 or n2 == 0:
            continue
        
        # Sum of differences between groups
        cross_diff = diff_matrix[np.ix_(side1.astype(bool), ~side1.astype(bool))].sum()
        
        count_pairs = n1 * n2
        if count_pairs == 0:
            continue
        
        avg_p = cross_diff / (count_pairs * L)
        avg_p = min(avg_p, 0.7499)
        
        d = -0.75 * np.log(1 - (4.0/3.0) * avg_p)
        total_score += (n1 * n2 / L) * d
    
    return total_score

def _dpp_refine(splits, arr, n, L, max_iterations=50):
    """
    Refine a set of splits using DPP-guided split exchanges.
    """
    if n < 4:
        return splits
    
    all_leaves = frozenset(range(n))
    current_splits = set(splits)
    current_score = _dpp_score_vectorized(current_splits, arr, n, L)
    
    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break
        
        improved = False
        # Try to add splits from a candidate pool
        # Generate candidate splits from random partitions
        for _ in range(20):
            if _time_left() < 5.0:
                break
            
            # Random balanced split
            k = np.random.randint(2, n-1)
            indices = np.random.choice(n, size=k, replace=False)
            new_split = frozenset(indices)
            other = frozenset(all_leaves - new_split)
            
            # Normalize: use the smaller side
            if len(new_split) > len(other):
                new_split = other
            
            if len(new_split) <= 1 or len(new_split) >= n-1:
                continue
            
            if new_split in current_splits:
                continue
            
            # Check compatibility with existing splits
            compatible = True
            for existing in current_splits:
                # Two splits are incompatible if they cross
                # i.e., there exist a,b in split1 and c,d in split2
                # such that a,c in one side and b,d in the other
                side1 = existing
                side2 = all_leaves - existing
                side3 = new_split
                side4 = all_leaves - new_split
                
                # Check all four combinations
                if (side1 & side3) and (side1 & side4) and (side2 & side3) and (side2 & side4):
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Try adding this split
            trial_splits = current_splits | {new_split}
            trial_score = _dpp_score_vectorized(trial_splits, arr, n, L)
            
            if trial_score < current_score:
                current_splits = trial_splits
                current_score = trial_score
                improved = True
        
        if not improved:
            break
    
    return current_splits

def _splits_to_tree(splits, labels):
    """Convert a set of splits to a Newick tree string."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    
    label_set = frozenset(labels)
    n = len(labels)
    
    # Filter splits to only those that partition the current label set
    valid_splits = []
    for s in splits:
        side1 = s & label_set
        side2 = label_set - side1
        if 1 < len(side1) < n and 1 < len(side2) < n:
            valid_splits.append(side1)
    
    if not valid_splits:
        return "(" + ",".join(labels) + ")"
    
    # Find the most balanced split
    best_split = None
    best_balance = 0
    for s in valid_splits:
        balance = min(len(s), n - len(s))
        if balance > best_balance:
            best_balance = balance
            best_split = s
    
    if best_split is None:
        return "(" + ",".join(labels) + ")"
    
    side1 = sorted(best_split, key=lambda x: int(x[1:]))
    side2 = sorted(label_set - best_split, key=lambda x: int(x[1:]))
    
    tree1 = _splits_to_tree(splits, side1)
    tree2 = _splits_to_tree(splits, side2)
    
    return f"({tree1},{tree2})"

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    # Compute distance matrix
    d_full = _jc_matrix(arr)
    
    # Initial NJ tree
    tree_full, splits_full = _nj(d_full, labels)
    
    # Bootstrap to get multiple trees
    num_bootstraps = 0
    max_bootstraps = min(30, max(5, L // 30))
    candidates = [(tree_full, splits_full)]
    
    while num_bootstraps < max_bootstraps and _time_left() > 15.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        num_bootstraps += 1
    
    # Score all candidates using DPP
    if len(candidates) > 1:
        best_idx = 0
        best_score = float('inf')
        for i, (tree_i, splits_i) in enumerate(candidates):
            score = _dpp_score_vectorized(splits_i, arr, n, L)
            if score < best_score:
                best_score = score
                best_idx = i
        
        # Also try consensus: keep splits that appear in >50% of trees
        split_count = {}
        for _, sp in candidates:
            for s in sp:
                split_count[s] = split_count.get(s, 0) + 1
        
        total_trees = len(candidates)
        consensus_splits = set()
        for s, count in split_count.items():
            if count > total_trees // 2:
                consensus_splits.add(s)
        
        # Score consensus
        if consensus_splits:
            consensus_score = _dpp_score_vectorized(consensus_splits, arr, n, L)
            if consensus_score < best_score:
                best_score = consensus_score
                # Use consensus splits
                tree_from_consensus = _splits_to_tree(consensus_splits, labels)
                return tree_from_consensus
        
        # Get the best candidate's splits and refine
        best_tree, best_splits = candidates[best_idx]
        
        # DPP refinement
        refined_splits = _dpp_refine(best_splits, arr, n, L, max_iterations=30)
        
        # Compare refined vs original
        refined_score = _dpp_score_vectorized(refined_splits, arr, n, L)
        if refined_score < best_score:
            tree_refined = _splits_to_tree(refined_splits, labels)
            return tree_refined
        else:
            return best_tree
    else:
        # Only one tree, still try to refine
        refined_splits = _dpp_refine(splits_full, arr, n, L, max_iterations=30)
        refined_score = _dpp_score_vectorized(refined_splits, arr, n, L)
        original_score = _dpp_score_vectorized(splits_full, arr, n, L)
        
        if refined_score < original_score:
            tree_refined = _splits_to_tree(refined_splits, labels)
            return tree_refined
        else:
            return tree_full
# EVOLVE-BLOCK-END
