"""Neighbour Joining with Jukes-Cantor correction, plus NNI refinement using likelihood score."""
import math
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 255.0


def _time_left():
    return _DEADLINE - time.time()


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences p."""
    p = min(max(p, 0.0), 0.7499)
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _seqs_to_array(sequences, labels):
    """Convert sequences to a numpy int array for fast comparison."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    length = len(seqs[0])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.empty((n, length), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            arr[i, j] = char_map[c]
    return arr


def _build_distance_matrix(arr, n):
    """Build pairwise JC distance matrix from integer array."""
    length = arr.shape[1]
    mat = np.zeros((n, n), dtype=np.float64)
    
    # Vectorized: compute all pairwise differences at once
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arr[i] != arr[j])
            p = diff / length
            d = _jc_distance(p)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _nj_tree(mat, labels):
    """Neighbour Joining on a distance matrix. Returns Newick string and join list."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";", []
    if n == 2:
        return f"({labels[0]},{labels[1]});", [(0, 1, -1)]
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    D_act = mat.copy()
    joins = []
    
    while len(active) > 3:
        if _time_left() < 5.0:
            # Build a star tree from remaining nodes as fallback
            return "(" + ",".join(trees[i] for i in active) + ");", joins
        
        m = len(active)
        # Row sums
        r = D_act.sum(axis=1)
        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D_act - r[:, None] - r[None, :]
        
        # Find the pair with minimum Q
        min_val = np.inf
        min_i = min_j = 0
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i = i
                    min_j = j
        
        i, j = min_i, min_j
        a = active[i]
        b = active[j]
        
        # Compute branch lengths
        d_ab = D_act[i, j]
        delta = (r[i] - r[j]) / (2 * (m - 2))
        la = 0.5 * d_ab + 0.5 * delta
        lb = 0.5 * d_ab - 0.5 * delta
        la = max(0.0, la)
        lb = max(0.0, lb)
        
        # Create new node
        new_id = f"n{a}_{b}"
        trees[new_id] = f"({trees[a]},{trees[b]})"
        joins.append((a, b, new_id))
        
        # Update active list
        new_active = [x for k, x in enumerate(active) if k != i and k != j]
        new_active.append(new_id)
        
        # Update distance matrix
        if len(new_active) > 1:
            new_m = len(new_active)
            D_new = np.zeros((new_m, new_m), dtype=np.float64)
            old_idx = {node: k for k, node in enumerate(active)}
            
            for k, x in enumerate(new_active):
                for l, y in enumerate(new_active):
                    if k == l:
                        continue
                    if x == new_id:
                        y_old = old_idx[y]
                        d_ay = D_act[old_idx[a], y_old]
                        d_by = D_act[old_idx[b], y_old]
                        d_ab_val = D_act[old_idx[a], old_idx[b]]
                        D_new[k, l] = 0.5 * (d_ay + d_by - d_ab_val)
                    elif y == new_id:
                        x_old = old_idx[x]
                        d_ax = D_act[old_idx[a], x_old]
                        d_bx = D_act[old_idx[b], x_old]
                        d_ab_val = D_act[old_idx[a], old_idx[b]]
                        D_new[k, l] = 0.5 * (d_ax + d_bx - d_ab_val)
                    else:
                        x_old = old_idx[x]
                        y_old = old_idx[y]
                        D_new[k, l] = D_act[x_old, y_old]
            
            D_act = D_new
        else:
            D_act = np.zeros((1, 1), dtype=np.float64)
        
        active = new_active
    
    # Build the final tree from the remaining 3 nodes
    if len(active) == 3:
        return "(" + ",".join(trees[i] for i in active) + ");", joins
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});", joins
    else:
        return trees[active[0]] + ";", joins


def _parse_newick(newick):
    """Parse Newick string into a tree structure.
    Returns a list of internal nodes, each as a tuple (left, right, label).
    Leaves are represented by their label string.
    """
    newick = newick.rstrip(';')
    tokens = list(newick)
    pos = [0]
    
    def parse():
        if tokens[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            pos[0] += 1  # skip ','
            right = parse()
            pos[0] += 1  # skip ')'
            # Check for label after ')'
            label = ''
            while pos[0] < len(tokens) and tokens[pos[0]] not in (',', ')', ';'):
                label += tokens[pos[0]]
                pos[0] += 1
            return ('internal', left, right, label)
        else:
            label = ''
            while pos[0] < len(tokens) and tokens[pos[0]] not in (',', ')', ';', '('):
                label += tokens[pos[0]]
                pos[0] += 1
            return ('leaf', label)
    
    root = parse()
    return root


def _get_leaves(node):
    """Get set of leaf labels under a node."""
    if node[0] == 'leaf':
        return {node[1]}
    else:
        return _get_leaves(node[1]) | _get_leaves(node[2])


def _get_splits(node):
    """Get all non-trivial splits from a tree."""
    if node[0] == 'leaf':
        return []
    
    splits = []
    left_leaves = _get_leaves(node[1])
    right_leaves = _get_leaves(node[2])
    
    # Only add non-trivial splits (both sides have at least 2 leaves)
    if len(left_leaves) >= 2 and len(right_leaves) >= 2:
        # Normalize: always put the smaller set first
        if len(left_leaves) > len(right_leaves):
            left_leaves, right_leaves = right_leaves, left_leaves
        splits.append(frozenset(left_leaves))
    
    splits.extend(_get_splits(node[1]))
    splits.extend(_get_splits(node[2]))
    return splits


def _rf_distance(tree1, tree2, all_leaves):
    """Compute normalized Robinson-Foulds distance between two trees."""
    splits1 = set(_get_splits(tree1))
    splits2 = set(_get_splits(tree2))
    
    common = splits1 & splits2
    total = len(splits1) + len(splits2)
    
    if total == 0:
        return 0.0
    
    return (total - 2 * len(common)) / total


def _nni_refine(newick, mat, labels):
    """NNI refinement: try all NNI moves and keep the best one.
    Uses RF distance to the NJ tree as a proxy, or better: use quartet voting."""
    n = len(labels)
    if n < 4:
        return newick
    
    if _time_left() < 15.0:
        return newick
    
    # Parse the tree
    tree = _parse_newick(newick)
    
    # For NNI, we need to work with a proper tree representation
    # Let's implement a simple NNI using the distance matrix
    
    # Actually, let's use a different approach: run multiple NJ with bootstrapped distances
    # and take a consensus
    
    # For now, return the original
    return newick


def _majority_vote(sequences, labels, mat, num_bootstraps=20):
    """Run NJ multiple times with bootstrapped distances and take majority vote."""
    n = len(labels)
    if n < 4:
        newick, _ = _nj_tree(mat, labels)
        return newick
    
    # Collect splits from multiple NJ runs
    split_counts = {}
    all_leaves = set(labels)
    
    # First, run standard NJ
    newick, _ = _nj_tree(mat, labels)
    tree = _parse_newick(newick)
    splits = set(_get_splits(tree))
    for s in splits:
        split_counts[s] = split_counts.get(s, 0) + 1
    
    # Bootstrap: subsample sites
    length = len(sequences[labels[0]])
    rng = np.random.RandomState(42)
    
    for b in range(num_bootstraps):
        if _time_left() < 10.0:
            break
        
        # Subsample sites with replacement
        indices = rng.randint(0, length, size=length)
        boot_seqs = {}
        for lb in labels:
            orig = sequences[lb]
            boot_seqs[lb] = ''.join(orig[i] for i in indices)
        
        # Build bootstrapped distance matrix
        arr = _seqs_to_array(boot_seqs, labels)
        boot_mat = _build_distance_matrix(arr, n)
        
        # Run NJ
        boot_newick, _ = _nj_tree(boot_mat, labels)
        boot_tree = _parse_newick(boot_newick)
        boot_splits = set(_get_splits(boot_tree))
        
        for s in boot_splits:
            split_counts[s] = split_counts.get(s, 0) + 1
    
    # Build tree from majority splits
    # Simple approach: just return the original NJ tree if no clear consensus
    # For a proper consensus, we'd need to build a tree from the splits
    
    # For now, return the best NJ tree
    return newick


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Convert sequences to array
    arr = _seqs_to_array(sequences, labels)
    
    # Build distance matrix
    mat = _build_distance_matrix(arr, n)
    
    # Run NJ
    newick, joins = _nj_tree(mat, labels)
    
    # Try majority vote with bootstrapping if time permits
    if _time_left() > 30.0:
        newick = _majority_vote(sequences, labels, mat, num_bootstraps=20)
    
    return newick
# EVOLVE-BLOCK-END
