"""Phylogenetic topology recovery: vectorized JC distances + Fast NJ for all n,
plus exhaustive quartet-score topology search for very small n (<=7)."""

import time
import math
import itertools
import numpy as np

_DEADLINE_MARGIN = 15.0
_START_TIME = None


def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _check_deadline():
    return _get_elapsed() > (270.0 - _DEADLINE_MARGIN)


def _encode_sequences(sequences, labels):
    """Convert dict of sequences to (n, L) uint8 array (A=0,C=1,G=2,T=3)."""
    n = len(labels)
    if n == 0:
        return np.empty((0, 0), dtype=np.uint8)
    L = len(sequences[labels[0]])
    if L == 0:
        return np.empty((n, 0), dtype=np.uint8)
    mat = np.empty((n, L), dtype=np.uint8)
    # Fast byte-to-nucleotide map
    # A=65, C=67, G=71, T=84
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        # Use translate for speed
        # Map any char not in ACGT to A (0)
        trans = str.maketrans('ACGT', 'ACGT')
        # Simpler: use numpy frombuffer + lookup
        arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        # Lookup: index by byte value
        # Build a 256-entry lookup
        # Do it once outside? For simplicity, inline small lookup
        # A=65->0, C=67->1, G=71->2, T=84->3
        # We'll use np.where chain (fast enough for L<=2000, n<=32)
        mat[i] = np.select(
            [arr == 65, arr == 67, arr == 71, arr == 84],
            [0, 1, 2, 3],
            default=0
        ).astype(np.uint8)
    return mat


def _jc_distance_matrix(seqs_mat, L):
    """Vectorized JC-corrected pairwise distance matrix."""
    n = seqs_mat.shape[0]
    if L == 0 or n == 0:
        return np.zeros((n, n), dtype=np.float64)
    # Mismatch counts
    diff = (seqs_mat[:, None, :] != seqs_mat[None, :, :])
    mismatches = diff.sum(axis=2)
    p = mismatches.astype(np.float64) / float(L)
    p = np.clip(p, 0.0, 0.75 - 1e-12)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def _fast_nj(D, labels):
    """Fast Neighbor-Joining on a distance matrix. Returns Newick string."""
    n = D.shape[0]
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]});"

    # active nodes: list of (name, newick_string)
    # We track by index into a growing list of subtrees
    # Simpler: maintain a list of node names and a distance matrix
    # Each node is either a leaf label or a composite string
    
    # Use a list-based approach
    names = list(labels)
    dist = D.copy()
    # Map from name -> newick string
    newick_map = {name: name for name in labels}
    
    while len(names) > 3:
        if _check_deadline():
            break
        m = len(names)
        # Compute Q matrix
        r = dist.sum(axis=1)
        Q = (m - 2) * dist - r[:, None] - r[None, :]
        # Find min Q (i != j)
        Q_masked = Q.copy()
        np.fill_diagonal(Q_masked, np.inf)
        idx = np.unravel_index(np.argmin(Q_masked), (m, m))
        i, j = idx[0], idx[1]
        if i == j:
            # Should not happen, but safety
            i, j = 0, 1
        
        a_name = names[i]
        b_name = names[j]
        d_ab = dist[i, j]
        
        # New node
        new_name = f"n{len(names)}_{a_name}_{b_name}"
        new_newick = f"({newick_map[a_name]},{newick_map[b_name]})"
        
        # Compute distances from new node to all remaining nodes
        rest = [k for k in range(m) if k not in (i, j)]
        if rest:
            d_a_r = dist[i, rest]
            d_b_r = dist[j, rest]
            d_new_r = 0.5 * (d_a_r + d_b_r - d_ab)
            # Build new dist matrix
            new_names = [names[k] for k in rest] + [new_name]
            new_dist = np.zeros((len(rest) + 1, len(rest) + 1), dtype=np.float64)
            # Copy distances among rest
            for idx_r, k in enumerate(rest):
                for idx_s, l in enumerate(rest):
                    if idx_r != idx_s:
                        new_dist[idx_r, idx_s] = dist[k, l]
            # Set distances to new node
            new_dist[:-1, -1] = d_new_r
            new_dist[-1, :-1] = d_new_r
            new_dist[-1, -1] = 0.0
            dist = new_dist
            names = new_names
        else:
            # Only 2 nodes left after removal? No, we remove 2 and add 1, so if m=4, rest has 2
            # If m=3, we break out of loop
            pass
        
        newick_map[new_name] = new_newick
    
    # Final tree with 3 nodes
    if len(names) == 3:
        a, b, c = names
        return f"(({newick_map[a]},{newick_map[b]}),{newick_map[c]});"
    elif len(names) == 2:
        a, b = names
        return f"({newick_map[a]},{newick_map[b]});"
    else:
        return newick_map[names[0]]


def _star(n, labels):
    return "(" + ",".join(labels) + ");"


def _quartet_score_tree(tree, labels, dist):
    """
    Compute total quartet score for a given tree topology.
    tree: nested tuple of ints (leaf indices)
    labels: list of taxon labels
    dist: (n, n) distance matrix
    Returns total score (lower is better).
    """
    n = len(labels)
    
    def _contains(node, target):
        if isinstance(node, int):
            return node == target
        for child in node:
            if _contains(child, target):
                return True
        return False
    
    def _same_clade(x, y, node):
        if isinstance(node, int):
            return node == x and node == y
        for child in node:
            if _contains(child, x) and _contains(child, y):
                return True
        return False
    
    total = 0.0
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        ab_same = _same_clade(a, b, tree)
        ac_same = _same_clade(a, c, tree)
        ad_same = _same_clade(a, d, tree)
        
        if ab_same and not ac_same and not ad_same:
            # (ab|cd)
            total += dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
        elif ac_same and not ab_same and not ad_same:
            # (ac|bd)
            total += dist[a, b] + dist[c, d] - dist[a, c] - dist[b, d]
        elif ad_same and not ab_same and not ac_same:
            # (ad|bc)
            total += dist[a, b] + dist[c, d] - dist[a, d] - dist[b, c]
    
    return total


def _generate_all_trees(n_leaves):
    """Generate all unrooted binary tree topologies as nested tuples of ints."""
    if n_leaves == 1:
        yield (0,)
    elif n_leaves == 2:
        yield ((0,), (1,))
    elif n_leaves == 3:
        yield ((0,), (1,), (2,))
    else:
        seen = set()
        for k in range(1, n_leaves // 2 + 1):
            for left in _generate_all_trees(k):
                right_indices = list(range(k, n_leaves))
                for right in _generate_all_trees(n_leaves - k):
                    def _remap(node, offset):
                        if isinstance(node, int):
                            return node + offset
                        return tuple(_remap(child, offset) for child in node)
                    
                    right_remapped = _remap(right, k)
                    tree = (left, right_remapped)
                    # Canonical form
                    canon = _canonicalize(tree)
                    if canon not in seen:
                        seen.add(canon)
                        yield tree


def _canonicalize(tree):
    if isinstance(tree, int):
        return tree
    children = [_canonicalize(child) for child in tree]
    children.sort(key=lambda x: repr(x))
    return tuple(children)


def _tree_to_newick(tree, labels):
    if isinstance(tree, int):
        return labels[tree]
    children = [_tree_to_newick(child, labels) for child in tree]
    return "(" + ",".join(children) + ")"


def _exhaustive_small_n(labels, dist):
    """Exhaustive search over all unrooted binary trees for n <= 7."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    best_tree = None
    best_score = float('inf')
    
    for tree in _generate_all_trees(n):
        if _check_deadline():
            break
        score = _quartet_score_tree(tree, labels, dist)
        if score < best_score:
            best_score = score
            best_tree = tree
    
    if best_tree is None:
        return None
    
    return _tree_to_newick(best_tree, labels)


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    # Encode sequences
    seqs_mat = _encode_sequences(sequences, labels)
    L = seqs_mat.shape[1] if seqs_mat.size > 0 else 0
    
    if L == 0:
        return _star(n, labels)
    
    # Compute JC distance matrix
    D = _jc_distance_matrix(seqs_mat, L)
    
    # Check if all distances are zero
    if not np.any(D):
        return _star(n, labels)
    
    # For small n (<=7), try exhaustive search
    if n <= 7 and not _check_deadline():
        result = _exhaustive_small_n(labels, D)
        if result is not None:
            return result
    
    # For larger n, use Fast NJ
    return _fast_nj(D, labels)
