"""Distance-proportional parsimony (DPP) inspired topology reconstruction."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.75:
        return 1.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _get_leaves(tree):
    """Get all leaf indices from a nested tuple tree."""
    if isinstance(tree, int):
        return [tree]
    return _get_leaves(tree[0]) + _get_leaves(tree[1])


def _find_all_edges(tree):
    """Return list of (path, child_subtree) for each edge in the tree."""
    edges = []
    def _traverse(node, path):
        if isinstance(node, int):
            return
        left, right = node
        edges.append((path + ('L',), left))
        edges.append((path + ('R',), right))
        _traverse(left, path + ('L',))
        _traverse(right, path + ('R',))
    _traverse(tree, ())
    return edges


def _replace_at_path(tree, path, new_subtree):
    """Replace the subtree at the given path with new_subtree."""
    if not path:
        return new_subtree
    first, rest = path[0], path[1:]
    left, right = tree
    if first == 'L':
        return (_replace_at_path(left, rest, new_subtree), right)
    else:
        return (left, _replace_at_path(right, rest, new_subtree))


def _path_distance(tree, i, j):
    """Compute path length (number of edges) between leaves i and j."""
    def _find_path(node, target):
        if node == target:
            return ()
        if isinstance(node, int):
            return None
        left, right = node
        pl = _find_path(left, target)
        if pl is not None:
            return ('L',) + pl
        pr = _find_path(right, target)
        if pr is not None:
            return ('R',) + pr
        return None
    
    pi = _find_path(tree, i)
    pj = _find_path(tree, j)
    if pi is None or pj is None:
        return float('inf')
    
    common = 0
    for a, b in zip(pi, pj):
        if a == b:
            common += 1
        else:
            break
    return len(pi) + len(pj) - 2 * common


def _score_tree(tree, dist, n):
    """Score a tree: sum of squared (observed - scaled_implied) distances."""
    leaves = _get_leaves(tree)
    if len(leaves) < 2:
        return 0.0
    
    implied_list = []
    observed_list = []
    for idx_i in range(len(leaves)):
        for idx_j in range(idx_i + 1, len(leaves)):
            li = leaves[idx_i]
            lj = leaves[idx_j]
            imp = _path_distance(tree, li, lj)
            obs = dist[li, lj]
            implied_list.append(imp)
            observed_list.append(obs)
    
    if not observed_list:
        return 0.0
    
    mean_obs = np.mean(observed_list)
    mean_imp = np.mean(implied_list)
    
    if mean_imp == 0:
        return 0.0
    
    scale = mean_obs / mean_imp
    score = sum((obs - imp * scale) ** 2 for obs, imp in zip(observed_list, implied_list))
    return score


def _place_taxon(tree, taxon, dist, n):
    """
    Try placing taxon on every edge, return the best tree.
    """
    edges = _find_all_edges(tree)
    best_score = float('inf')
    best_tree = None
    
    for path, child in edges:
        # Create new subtree: (child, taxon)
        new_subtree = (child, taxon)
        new_tree = _replace_at_path(tree, path, new_subtree)
        
        # Score
        s = _score_tree(new_tree, dist, n)
        if s < best_score:
            best_score = s
            best_tree = new_tree
    
    return best_tree


def _to_newick(tree, labels):
    """Convert nested tuple tree to Newick string."""
    if isinstance(tree, int):
        return labels[tree]
    return f"({_to_newick(tree[0], labels)},{_to_newick(tree[1], labels)})"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};\n"
    if n == 2:
        return f"({labels[0]},{labels[1]});\n"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]});\n"
    
    dist = _compute_distance_matrix(sequences, labels)
    
    # Choose the 3 taxa with smallest total pairwise distance as starting triplet
    best3 = None
    best3_score = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = dist[i, j] + dist[j, k] + dist[i, k]
                if s < best3_score:
                    best3_score = s
                    best3 = (i, j, k)
    
    i, j, k = best3
    tree = ((i, j), k)
    
    # Add remaining taxa one by one
    remaining = [t for t in range(n) if t not in (i, j, k)]
    for taxon in remaining:
        tree = _place_taxon(tree, taxon, dist, n)
    
    return _to_newick(tree, labels) + ";\n"


# EVOLVE-BLOCK-END
