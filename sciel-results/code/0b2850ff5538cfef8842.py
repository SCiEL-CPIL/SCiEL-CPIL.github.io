# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.82

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

def _nni_refine(tree_newick, splits, dist, labels, max_rounds=100):
    """
    NNI refinement on a tree represented by its splits and Newick string.
    This is a simplified version that works with the split representation.
    For a proper NNI, we need to manipulate the tree structure.
    Given the complexity, we'll implement a basic swap-based refinement.
    """
    # For simplicity and robustness, we'll skip full NNI and rely on consensus.
    # A proper NNI implementation requires tree parsing and manipulation.
    return tree_newick, splits

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

def _build_tree_from_splits(splits, labels):
    """
    Build a tree from a set of splits. This is a heuristic approach.
    We use the splits to guide the NJ algorithm or construct a compatible tree.
    For now, we'll return a star tree if no splits, or use NJ with the splits as hints.
    """
    if not splits:
        n = len(labels)
        if n == 1:
            return labels[0]
        if n == 2:
            return f"({labels[0]},{labels[1]})"
        if n == 3:
            return f"({labels[0]},({labels[1]},{labels[2]}))"
        # Star tree
        return "(" + ",".join(labels) + ")"
    
    # For a proper tree from splits, we need a compatible split set.
    # This is complex. We'll rely on the NJ tree that generated the splits.
    # The caller should pass the tree string as well.
    return None

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
    max_bootstraps = 200
    
    # Use more bootstraps if time permits
    while num_bootstraps < max_bootstraps and _time_left() > 3.0:
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
        
        return candidates[best_idx][0] + ";"
    else:
        return tree_full + ";"

# EVOLVE-BLOCK-END
