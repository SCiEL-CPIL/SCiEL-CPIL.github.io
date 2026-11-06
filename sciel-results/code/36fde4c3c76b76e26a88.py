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
    d = -0.75 * np.log(1 - (4.0 / 3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d


def _nj(dist, labels):
    """Neighbor-joining tree reconstruction. Returns Newick string and set of splits."""
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
            for b in nodes[i + 1:]:
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


def _dpp_score(dist, labels, splits):
    """
    Compute the Distance-Proportional Parsimony (DPP) score for a tree topology.
    The DPP score is the sum over all pairs of taxa of min(distance on one side, distance on other side)
    where the split separates the two taxa.
    
    For a given split S (subset of taxa), for each pair (a,b) where a in S and b not in S:
    the contribution is min(dist[a][b], ...) - actually DPP uses the distance matrix directly.
    
    Standard DPP: For a tree T with splits, the DPP score = sum over all unordered pairs {i,j}
    of d(i,j) where the path between i and j crosses the split. But the standard formulation is:
    
    DPP(T) = sum over splits S of T of sum_{i in S, j not in S} d(i,j) / (n-1)
    
    Actually, let me use the correct DPP definition:
    DPP(T) = sum over all internal edges (splits) of the total distance across that split.
    """
    n = len(labels)
    if n < 3:
        return 0.0

    all_leaves = frozenset(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    score = 0.0
    for split in splits:
        other = frozenset(all_leaves - split)
        for a in split:
            for b in other:
                score += dist[label_to_idx[a]][label_to_idx[b]]
    return score


def _dpp_refine(dist, labels, initial_tree_str, initial_splits, max_rounds=5):
    """
    Refine tree topology using DPP-guided NNI moves.
    Try NNI moves and accept if DPP score improves.
    """
    n = len(labels)
    if n < 4:
        return initial_tree_str, initial_splits

    all_leaves = frozenset(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    current_splits = set(initial_splits)
    current_score = _dpp_score(dist, labels, current_splits)
    best_splits = set(current_splits)
    best_score = current_score
    best_tree = initial_tree_str

    for _round in range(max_rounds):
        if _time_left() < 5.0:
            break

        # Build current tree structure for NNI
        tree, splits = _parse_tree_to_struct(initial_tree_str, labels)
        if tree is None:
            break

        improved = False
        for _inner in range(min(50, 2 * n)):
            if _time_left() < 3.0:
                break

            # Pick a random internal edge to try NNI on
            # Represent tree as nested tuples: (left, right) for internal, label for leaf
            def pick_random_internal(node, path=[]):
                """Collect all internal node positions."""
                results = []
                if isinstance(node, tuple) and len(node) == 2 and not isinstance(node[0], str):
                    results.append((node, list(path)))
                    results.extend(pick_random_internal(node[0], path + [0]))
                    results.extend(pick_random_internal(node[1], path + [1]))
                elif isinstance(node, tuple) and len(node) == 2:
                    # Check if children are internal
                    if not isinstance(node[0], str) and not isinstance(node[1], str):
                        results.append((node, list(path)))
                    results.extend(pick_random_internal(node[0], path + [0]))
                    results.extend(pick_random_internal(node[1], path + [1]))
                return results

            def get_leaves(node):
                if isinstance(node, str):
                    return frozenset([node])
                elif isinstance(node, tuple):
                    result = frozenset()
                    for child in node:
                        result = result | get_leaves(child)
                    return result
                return frozenset()

            def build_tree_str(node):
                if isinstance(node, str):
                    return node
                elif isinstance(node, tuple):
                    return "(" + build_tree_str(node[0]) + "," + build_tree_str(node[1]) + ")"
                return ""

            def parse_to_struct(s, pos=0):
                if s[pos] == '(':
                    pos += 1
                    left, pos = parse_to_struct(s, pos)
                    # expect comma
                    if s[pos] == ',':
                        pos += 1
                    right, pos = parse_to_struct(s, pos)
                    if s[pos] == ')':
                        pos += 1
                    return (left, right), pos
                else:
                    end = pos
                    while end < len(s) and s[end] not in ',)':
                        end += 1
                    label = s[pos:end].strip()
                    return label, end

            tree_struct, _ = parse_to_struct(initial_tree_str)

            def apply_nni(node, path):
                """Apply NNI at the given path. Returns new tree struct."""
                if not path:
                    return node
                idx = path[0]
                child = node[idx]
                if not isinstance(child, tuple):
                    return node
                # child is (left, right), try the two alternative topologies
                # Current: ((a,b),(c,d))
                # Alt 1: ((a,c),(b,d))
                # Alt 2: ((a,d),(b,c))
                a, b = child[0], child[1]
                # Get the sibling
                sibling = node[1 - idx]

                # Try alternative 1: (a, sibling), (b, c) where c is from the other side
                # Actually NNI: for edge (a,b)|(c,d), alternatives are (a,c)|(b,d) and (a,d)|(b,c)
                # Here node is (sibling, child) or (child, sibling)
                # child = (x, y), sibling = z
                # Current: (z, (x,y))
                # Alt 1: ((z, x), y)
                # Alt 2: ((z, y), x)

                alt1 = ((sibling, a), b)
                alt2 = ((sibling, b), a)

                # Compute splits for each alternative
                def compute_splits_from_struct(t):
                    splits_set = set()

                    def traverse(node_t):
                        if not isinstance(node_t, tuple):
                            return
                        leaves_t = get_leaves(node_t)
                        if 1 < len(leaves_t) < n:
                            side1 = leaves_t
                            side2 = frozenset(all_leaves - leaves_t)
                            if tuple(sorted(side1)) < tuple(sorted(side2)):
                                splits_set.add(side1)
                            else:
                                splits_set.add(side2)
                        for child_t in node_t:
                            traverse(child_t)

                    traverse(t)
                    return splits_set

                new_tree_alt1 = tuple(alt1) if path == [0] or path == [1] else None

                # Rebuild tree with NNI applied
                def rebuild_with_nni(node, path, alt):
                    if not path:
                        if alt is None:
                            return node
                        return alt
                    idx = path[0]
                    new_child = rebuild_with_nni(node[idx], path[1:], alt) if len(path) > 1 else (node[0], node[1])
                    if len(path) == 1:
                        new_node = list(node)
                        new_node[idx] = alt
                        return tuple(new_node)
                    else:
                        new_node = list(node)
                        new_node[idx] = new_child
                        return tuple(new_node)

                # Simpler: just rebuild the whole tree
                def tree_with_nni(tree_root, path, alt_choice):
                    """alt_choice: 0=no change, 1=swap to alt1, 2=swap to alt2"""
                    if not path:
                        return tree_root
                    idx = path[0]
                    child = tree_root[idx]
                    if not isinstance(child, tuple):
                        return tree_root
                    if len(path) == 1:
                        a, b = child[0], child[1]
                        sibling = tree_root[1 - idx]
                        if alt_choice == 1:
                            new_child = ((sibling, a), b)
                        elif alt_choice == 2:
                            new_child = ((sibling, b), a)
                        else:
                            new_child = child
                        new_node = list(tree_root)
                        new_node[idx] = new_child
                        return tuple(new_node)
                    else:
                        new_child = tree_with_nni(child, path[1:], alt_choice)
                        new_node = list(tree_root)
                        new_node[idx] = new_child
                        return tuple(new_node)

                # Find a random internal edge path
                def find_internal_paths(node, path=None):
                    if path is None:
                        path = []
                    paths = []
                    if isinstance(node, tuple):
                        if not isinstance(node[0], str) and not isinstance(node[1], str):
                            # Both children are internal or at least one is
                            paths.append(list(path))
                        if isinstance(node[0], tuple):
                            paths.extend(find_internal_paths(node[0], path + [0]))
                        if isinstance(node[1], tuple):
                            paths.extend(find_internal_paths(node[1], path + [1]))
                    return paths

                internal_paths = find_internal_paths(tree_struct)
                if not internal_paths:
                    break

                chosen_path = internal_paths[_inner % len(internal_paths)]
                for alt_choice in [1, 2]:
                    new_tree_struct = tree_with_nni(tree_struct, chosen_path, alt_choice)
                    new_splits = compute_splits_from_struct(new_tree_struct)
                    new_score = _dpp_score(dist, labels, new_splits)
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        current_splits = new_splits
                        # Update best
                        new_tree_str = build_tree_str(new_tree_struct)
                        if new_score < best_score:
                            best_score = new_score
                            best_splits = set(current_splits)
                            best_tree = new_tree_str
                        # Also update the working tree
                        initial_tree_str = new_tree_str
                        improved = True
                        break
                if not improved and _inner >= 20:
                    break

        if not improved:
            break

    return best_tree, best_splits


def _parse_tree_to_struct(tree_str, labels):
    """Parse Newick string into nested tuple structure."""
    if not tree_str or not tree_str.strip():
        return None, None

    s = tree_str.strip().rstrip(';')

    def parse(s, pos=0):
        if s[pos] == '(':
            pos += 1
            left, pos = parse(s, pos)
            if pos < len(s) and s[pos] == ',':
                pos += 1
            right, pos = parse(s, pos)
            if pos < len(s) and s[pos] == ')':
                pos += 1
            return (left, right), pos
        else:
            end = pos
            while end < len(s) and s[end] not in ',)':
                end += 1
            label = s[pos:end].strip()
            return label, end

    tree_struct, _ = parse(s)
    return tree_struct, None


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

    # For small n, do DPP refinement with multiple restarts
    if n <= 20:
        # DPP refine the NJ tree
        refined_tree, refined_splits = _dpp_refine(d_full, labels, tree_full, splits_full, max_rounds=3)
        return refined_tree
    else:
        # For larger n, use bootstrap consensus
        num_bootstraps = 0
        max_bootstraps = min(30, max(8, L // 25))
        candidates = [(tree_full, splits_full)]

        np.random.seed(42)
        while num_bootstraps < max_bootstraps and _time_left() > 15.0:
            indices = np.random.randint(0, L, L)
            arr_boot = arr[:, indices]
            d_boot = _jc_matrix(arr_boot)
            tree_boot, splits_boot = _nj(d_boot, labels)
            candidates.append((tree_boot, splits_boot))
            num_bootstraps += 1

        # Find consensus: keep splits appearing in >60% of trees
        split_count = {}
        for _, sp in candidates:
            for s in sp:
                split_count[s] = split_count.get(s, 0) + 1

        total_trees = len(candidates)
        consensus_splits = set()
        for s, count in split_count.items():
            if count >= total_trees * 0.6:
                consensus_splits.add(s)

        # Return the tree whose splits are closest to consensus
        if consensus_splits:
            best_idx = 0
            min_dist = float('inf')
            for i, (tree_i, splits_i) in enumerate(candidates):
                dist = len(splits_i.symmetric_difference(consensus_splits))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            return candidates[best_idx][0]
        else:
            return tree_full

# EVOLVE-BLOCK-END
