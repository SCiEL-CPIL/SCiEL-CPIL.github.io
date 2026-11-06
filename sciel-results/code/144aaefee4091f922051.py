# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

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

def _quartet_ml_scores(arr, i, j, k, l):
    """
    Compute ML scores for the three possible quartet topologies
    for taxa i,j,k,l under Jukes-Cantor model.
    Returns (score_ijk|kl, score_ikj|lk, score_ij|kl)
    where score is log-likelihood (higher is better).
    """
    L = arr.shape[1]
    
    # Get sequences for these 4 taxa
    si = arr[i]
    sj = arr[j]
    sk = arr[k]
    sl = arr[l]
    
    # Under JC model, the three topologies are:
    # T1: ((i,j),(k,l)) - i,j sister; k,l sister
    # T2: ((i,k),(j,l)) - i,k sister; j,l sister  
    # T3: ((i,l),(j,k)) - i,l sister; j,k sister
    
    # For JC model, we can compute likelihood by considering
    # the pattern of differences at each site.
    
    # Precompute pairwise difference indicators
    dij = si != sj
    dik = si != sk
    dil = si != sl
    djk = sj != sk
    djlk = sj != sl
    dkll = sk != sl
    
    # For JC model with 4 taxa, the likelihood at each site depends on
    # the pattern of which taxa have which bases.
    # We can use the fact that under JC, P(same) = 1/4 + 3/4 * exp(-4t/3)
    # and P(diff) = 3/4 * (1 - exp(-4t/3))/3
    
    # For simplicity and speed, use a heuristic based on pairwise distances:
    # The best topology is the one where the sum of distances across the split
    # is minimized (this is equivalent to the quartet criterion).
    
    # T1: ((i,j),(k,l)): cross distances are d(i,k)+d(i,l)+d(j,k)+d(j,l)
    # T2: ((i,k),(j,l)): cross distances are d(i,j)+d(i,l)+d(k,j)+d(k,l)  
    # T3: ((i,l),(j,k)): cross distances are d(i,j)+d(i,k)+d(l,j)+d(l,k)
    
    # Use the number of differing sites as proxy for distance
    # T1 score: minimize d(i,k)+d(i,l)+d(j,k)+d(j,l)
    score_t1 = np.sum(dik) + np.sum(dil) + np.sum(djk) + np.sum(djlk)
    # T2 score: minimize d(i,j)+d(i,l)+d(k,j)+d(k,l)
    score_t2 = np.sum(dij) + np.sum(dil) + np.sum(djk) + np.sum(dkll)
    # T3 score: minimize d(i,j)+d(i,k)+d(l,j)+d(l,k)
    score_t3 = np.sum(dij) + np.sum(dik) + np.sum(djlk) + np.sum(dkll)
    
    # Lower score = better topology
    return (-score_t1, -score_t2, -score_t3)  # Return as log-likelihoods (higher better)

def _quartet_weighted_tree(labels, arr, max_quartets=None):
    """
    Build a tree using weighted quartet voting.
    For each quartet, find the best topology and accumulate weights.
    Then resolve the full tree using these quartet constraints.
    """
    n = len(labels)
    L = arr.shape[1]
    
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    # For large n, we can't evaluate all quartets (O(n^4))
    # Use a subset strategy: evaluate quartets from a random sample
    # or use a greedy approach
    
    # Strategy: Use NJ as initial guide, then refine with quartets
    # First, get an initial tree from NJ
    d = _jc_matrix(arr)
    nj_tree, nj_splits = _nj(d, labels)
    
    # For the quartet approach, we'll use a different strategy:
    # 1. Get candidate splits from NJ
    # 2. For each candidate split, evaluate its support using quartets
    # 3. Build tree from best-supported splits
    
    # Alternative: Use a recursive quartet-based tree building
    # This is more complex, so let's use a hybrid approach:
    # Use NJ tree as backbone, then try to improve using quartet information
    
    # For now, return the NJ tree as baseline
    # The quartet scoring will be used to validate/improve splits
    
    return nj_tree, nj_splits

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

def _quartet_support_score(split, labels, arr, max_samples=200):
    """
    Compute support for a split using quartet analysis.
    For each quartet that is compatible with the split, check if
    the quartet's best topology is also compatible.
    Returns a score in [0, 1] where 1 means full support.
    """
    n = len(labels)
    L = arr.shape[1]
    
    if len(split) <= 1 or len(split) >= n - 1:
        return 0.5
    
    label_to_idx = {l: i for i, l in enumerate(labels)}
    split_set = set(split)
    
    # Sample quartets that cross the split (have 2 on each side)
    side1 = [l for l in labels if l in split_set]
    side2 = [l for l in labels if l not in split_set]
    
    if len(side1) < 2 or len(side2) < 2:
        return 0.5
    
    # Sample up to max_samples quartets with 2 from each side
    np.random.seed(42)
    n_quartets = min(max_samples, len(side1) * len(side2) // 4)
    
    if n_quartets == 0:
        return 0.5
    
    compatible_count = 0
    total_count = 0
    
    for _ in range(n_quartets):
        # Pick 2 from side1 and 2 from side2
        i1, i2 = np.random.choice(len(side1), 2, replace=False)
        j1, j2 = np.random.choice(len(side2), 2, replace=False)
        
        a, b = side1[i1], side1[i2]
        c, d = side2[j1], side2[j2]
        
        ai, bi, ci, di = label_to_idx[a], label_to_idx[b], label_to_idx[c], label_to_idx[d]
        
        # Get quartet ML scores
        scores = _quartet_ml_scores(arr, ai, bi, ci, di)
        # scores[0] = ((a,b),(c,d)) - compatible with split
        # scores[1] = ((a,c),(b,d)) - incompatible
        # scores[2] = ((a,d),(b,c)) - incompatible
        
        best_topology = np.argmax(scores)
        
        # Check if best topology is compatible with the split
        # ((a,b),(c,d)) is compatible: a,b on one side, c,d on other
        # ((a,c),(b,d)) is incompatible: a,c on one side, b,d on other
        # ((a,d),(b,c)) is incompatible: a,d on one side, b,c on other
        
        if best_topology == 0:
            compatible_count += 1
        
        total_count += 1
    
    if total_count == 0:
        return 0.5
    
    return compatible_count / total_count

def _quartet_refine_tree(tree_str, labels, arr, max_iterations=10):
    """
    Refine a tree by evaluating split support using quartet analysis.
    Remove poorly supported splits and try to add better ones.
    """
    n = len(labels)
    L = arr.shape[1]
    
    if n < 4:
        return tree_str
    
    # Get current splits
    current_splits = _get_splits_from_tree(tree_str, labels)
    
    # Evaluate support for each split
    split_support = {}
    for split in current_splits:
        support = _quartet_support_score(split, labels, arr, max_samples=100)
        split_support[split] = support
    
    # Keep only splits with reasonable support (> 0.5)
    refined_splits = set()
    for split, support in split_support.items():
        if support >= 0.5:
            refined_splits.add(split)
    
    # Build tree from refined splits
    if not refined_splits:
        return tree_str
    
    return _splits_to_tree(refined_splits, labels)

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
                side1 = existing
                side2 = all_leaves - existing
                side3 = new_split
                side4 = all_leaves - new_split
                
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
                # Try quartet refinement
                if _time_left() > 10.0:
                    refined_tree = _quartet_refine_tree(tree_from_consensus, labels, arr)
                    return refined_tree
                return tree_from_consensus
        
        # Get the best candidate's splits and refine
        best_tree, best_splits = candidates[best_idx]
        
        # DPP refinement
        refined_splits = _dpp_refine(best_splits, arr, n, L, max_iterations=30)
        
        # Compare refined vs original
        refined_score = _dpp_score_vectorized(refined_splits, arr, n, L)
        if refined_score < best_score:
            tree_refined = _splits_to_tree(refined_splits, labels)
            # Try quartet refinement
            if _time_left() > 10.0:
                final_tree = _quartet_refine_tree(tree_refined, labels, arr)
                return final_tree
            return tree_refined
        else:
            # Try quartet refinement on original
            if _time_left() > 10.0:
                final_tree = _quartet_refine_tree(best_tree, labels, arr)
                return final_tree
            return best_tree
    else:
        # Only one tree, still try to refine
        refined_splits = _dpp_refine(splits_full, arr, n, L, max_iterations=30)
        refined_score = _dpp_score_vectorized(refined_splits, arr, n, L)
        original_score = _dpp_score_vectorized(splits_full, arr, n, L)
        
        if refined_score < original_score:
            tree_refined = _splits_to_tree(refined_splits, labels)
            # Try quartet refinement
            if _time_left() > 10.0:
                final_tree = _quartet_refine_tree(tree_refined, labels, arr)
                return final_tree
            return tree_refined
        else:
            # Try quartet refinement on original
            if _time_left() > 10.0:
                final_tree = _quartet_refine_tree(tree_full, labels, arr)
                return final_tree
            return tree_full
# EVOLVE-BLOCK-END
