# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys

_START = None
_DEADLINE_FRACTION = 0.85 # Stop at 85% of 270s to be safe

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
    # Using uint8 for compactness
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    # Calculate pairwise differences efficiently
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
    
    # Copy dist to dict for easy access/modification
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
        
        # Record split
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        # Canonical form: lexicographically smaller frozenset
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

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    candidates = [(tree_full, splits_full)]
    split_counts = {}
    
    # Count splits in the first tree (NJ)
    for s in splits_full:
        split_counts[s] = split_counts.get(s, 0) + 1
        
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 2000 # Increased significantly
    
    # Time budget: 270s total.
    # We should leave 5s for cleanup and selection.
    # Target time for bootstraps: 265s.
    
    while num_bootstraps < max_bootstraps and _time_left() > 5.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        
        # Update split counts
        for s in splits_boot:
            split_counts[s] = split_counts.get(s, 0) + 1
            
        num_bootstraps += 1
        
    if len(candidates) > 1:
        total_bootstraps = len(candidates)
        threshold = 0.5 * total_bootstraps
        
        # Identify strong splits
        strong_splits = set()
        for s, c in split_counts.items():
            if c > threshold:
                strong_splits.add(s)
                
        best_idx = 0
        best_score = -1
        second_best_score = float('inf') # RF to NJ for tie-breaking
        
        # Pre-compute RF to NJ (index 0) for all candidates?
        # Or just compute on the fly if needed for tie-breaking.
        # Let's compute "number of strong splits in candidate"
        
        for i in range(len(candidates)):
            splits_i = candidates[i][1]
            # Count how many splits in splits_i are in strong_splits
            common = len(splits_i & strong_splits)
            
            if common > best_score:
                best_score = common
                best_idx = i
                # Recalculate tie-breaker only if we find a better score?
                # Or just store it.
                second_best_score = len(splits_i.symmetric_difference(candidates[0][1]))
            elif common == best_score:
                # Tie-break: minimize RF distance to NJ tree (index 0)
                rf_to_nj = len(splits_i.symmetric_difference(candidates[0][1]))
                if rf_to_nj < second_best_score:
                    best_idx = i
                    second_best_score = rf_to_nj
                    
        return candidates[best_idx][0]
    else:
        return tree_full

# EVOLVE-BLOCK-END
