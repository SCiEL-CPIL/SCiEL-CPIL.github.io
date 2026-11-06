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
    Lower is better. Uses likelihood-based scoring for each split.
    """
    if not splits or n < 3:
        return 0.0
    
    # Precompute pairwise difference matrix
    diff_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    
    total_score = 0.0
    for split in splits:
        side1_mask = np.zeros(n, dtype=bool)
        for l in split:
            idx = int(l[1:]) if l[0] == 't' else int(l)
            if idx < n:
                side1_mask[idx] = True
        n1 = side1_mask.sum()
        n2 = n - n1
        if n1 == 0 or n2 == 0:
            continue
        
        # Sum of differences between groups
        cross_diff = diff_matrix[np.ix_(side1_mask, ~side1_mask)].sum()
        
        count_pairs = n1 * n2
        if count_pairs == 0:
            continue
        
        avg_p = cross_diff / (count_pairs * L)
        avg_p = min(avg_p, 0.7499)
        
        # JC distance
        d = -0.75 * np.log(1 - (4.0/3.0) * avg_p)
        
        # Weight by number of pairs and normalize
        total_score += (n1 * n2 / L) * d
    
    return total_score

def _splits_compatible(s1, s2, all_leaves):
    """Check if two splits are compatible (don't cross)."""
    side1 = s1
    side2 = all_leaves - s1
    side3 = s2
    side4 = all_leaves - s2
    
    # Incompatible if all four intersections are non-empty
    return not ((side1 & side3) and (side1 & side4) and (side2 & side3) and (side2 & side4))

def _dpp_refine(splits, arr, n, L, max_iterations=50):
    """
    Refine a set of splits using DPP-guided split exchanges.
    Uses cherry-picking and more systematic candidate generation.
    """
    if n < 4:
        return splits
    
    # Convert to integer indices for efficiency
    def to_idx(f):
        return frozenset(int(x[1:]) if x[0] == 't' else int(x) for x in f)
    
    all_leaves = frozenset(range(n))
    current_splits = set(to_idx(s) for s in splits)
    current_score = _dpp_score_vectorized(set(frozenset(f"t{i}" for i in s) for s in current_splits), arr, n, L)
    
    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break
        
        improved = False
        
        # Phase 1: Try to add compatible splits
        for _ in range(30):
            if _time_left() < 5.0:
                break
            
            # Generate candidate split - try different strategies
            strategy = np.random.random()
            
            if strategy < 0.3:
                # Balanced random split
                k = np.random.randint(2, n-1)
                indices = np.random.choice(n, size=k, replace=False)
                new_split = frozenset(indices)
            elif strategy < 0.6:
                # Near-balanced split
                k = n // 2 + np.random.randint(-2, 3)
                k = max(2, min(n-2, k))
                indices = np.random.choice(n, size=k, replace=False)
                new_split = frozenset(indices)
            else:
                # Cherry-based: pick two random taxa, try to find a split separating them
                a, b = np.random.choice(n, size=2, replace=False)
                # Find a random third taxon to help define the split
                others = [i for i in range(n) if i != a and i != b]
                if not others:
                    continue
                c = np.random.choice(others)
                # Try splits that separate {a,b} from rest, or {a} from rest, etc.
                candidates = [
                    frozenset([a, b]),
                    frozenset([a]),
                    frozenset([b]),
                    frozenset([a, c]),
                    frozenset([b, c]),
                ]
                new_split = np.random.choice(len(candidates))
                new_split = candidates[new_split]
            
            other = all_leaves - new_split
            
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
                if not _splits_compatible(existing, new_split, all_leaves):
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Try adding this split
            trial_splits = current_splits | {new_split}
            trial_score = _dpp_score_vectorized(set(frozenset(f"t{i}" for i in s) for s in trial_splits), arr, n, L)
            
            if trial_score < current_score:
                current_splits = trial_splits
                current_score = trial_score
                improved = True
        
        # Phase 2: Try removing poorly scoring splits
        if len(current_splits) > 2:
            for existing in list(current_splits):
                if _time_left() < 5.0:
                    break
                
                trial_splits = current_splits - {existing}
                trial_score = _dpp_score_vectorized(set(frozenset(f"t{i}" for i in s) for s in trial_splits), arr, n, L)
                
                if trial_score < current_score:
                    current_splits = trial_splits
                    current_score = trial_score
                    improved = True
        
        if not improved:
            break
    
    # Convert back to label format
    return set(frozenset(f"t{i}" for i in s) for s in current_splits)

def _splits_to_tree(splits, labels):
    """Convert a set of splits to a Newick tree string using most balanced split first."""
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
    best_tree = tree_full
    best_splits = splits_full
    best_score = _dpp_score_vectorized(best_splits, arr, n, L)
    
    if len(candidates) > 1:
        for i, (tree_i, splits_i) in enumerate(candidates[1:]):
            score = _dpp_score_vectorized(splits_i, arr, n, L)
            if score < best_score:
                best_score = score
                best_tree = tree_i
                best_splits = splits_i
        
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
                best_splits = consensus_splits
                best_tree = _splits_to_tree(consensus_splits, labels)
    
    # DPP refinement on the best candidate
    if _time_left() > 10.0:
        refined_splits = _dpp_refine(best_splits, arr, n, L, max_iterations=40)
        refined_score = _dpp_score_vectorized(refined_splits, arr, n, L)
        
        if refined_score < best_score:
            best_tree = _splits_to_tree(refined_splits, labels)
            best_splits = refined_splits
            best_score = refined_score
    
    return best_tree
# EVOLVE-BLOCK-END
