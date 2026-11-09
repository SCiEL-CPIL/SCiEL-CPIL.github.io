"""Phylogenetic topology recovery: NJ + NNI refinement with quartet scores."""
import math
import time
import numpy as np

_DEADLINE = 250.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _jc_distance_matrix(seqs):
    n = len(seqs)
    L = len(seqs[0])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    M = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            M[i, j] = char_map[c]
    diff = (M[:, np.newaxis, :] != M[np.newaxis, :, :])
    p = diff.sum(axis=2) / L
    np.fill_diagonal(p, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(np.maximum(1.0 - (4.0/3.0) * p, 1e-10))
    d = np.where(p >= 0.7499, 3.0, d)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_build(D):
    n = D.shape[0]
    if n <= 2:
        return None
    dist = D.astype(float).copy()
    nodes = list(range(n))
    trees = {i: str(i) for i in range(n)}
    while len(nodes) > 3:
        m = len(nodes)
        row_sum = np.zeros(m)
        for idx, i in enumerate(nodes):
            for jdx, j in enumerate(nodes):
                if i != j:
                    row_sum[idx] += dist[i, j]
        best_c = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i+1, m):
                c = (m - 2) * dist[nodes[i], nodes[j]] - row_sum[i] - row_sum[j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
        a, b = nodes[best_i], nodes[best_j]
        new_node = max(nodes) + 1
        trees[new_node] = f"({trees[a]},{trees[b]})"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            dist[new_node, x] = dist[x, new_node] = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
        nodes = rest + [new_node]
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _parse_newick(s):
    s = s.strip().rstrip(';')
    if s.startswith('('):
        depth = 0
        end = -1
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        inner = s[1:end]
        children = []
        depth = 0
        start = 0
        for i, c in enumerate(inner):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                children.append(inner[start:i])
                start = i + 1
        children.append(inner[start:])
        return tuple(_parse_newick(c) for c in children)
    else:
        return int(s)


def _to_newick(tree):
    if isinstance(tree, int):
        return str(tree)
    return "(" + ",".join(_to_newick(c) for c in tree) + ")"


def _get_taxa(node):
    if isinstance(node, int):
        return {node}
    return _get_taxa(node[0]) | _get_taxa(node[1])


def _quartet_tree_score(D, tree, n):
    splits = set()
    def _get_splits(node):
        if isinstance(node, int):
            return
        left_taxa = _get_taxa(node[0])
        right_taxa = _get_taxa(node[1])
        splits.add(frozenset([frozenset(left_taxa), frozenset(right_taxa)]))
        _get_splits(node[0])
        _get_splits(node[1])
    _get_splits(tree)
    
    total = 0.0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    s1 = frozenset([frozenset([a,b]), frozenset([c,d])])
                    s2 = frozenset([frozenset([a,c]), frozenset([b,d])])
                    s3 = frozenset([frozenset([a,d]), frozenset([b,c])])
                    if s1 in splits:
                        total += D[a,b] + D[c,d]
                    elif s2 in splits:
                        total += D[a,c] + D[b,d]
                    elif s3 in splits:
                        total += D[a,d] + D[b,c]
    return total


def _replace_subtree(tree, old_node, new_node):
    if tree == old_node:
        return new_node
    if isinstance(tree, int):
        return None
    left, right = tree[0], tree[1]
    new_left = _replace_subtree(left, old_node, new_node)
    if new_left is not None:
        return (new_left, right)
    new_right = _replace_subtree(right, old_node, new_node)
    if new_right is not None:
        return (left, new_right)
    return None


def _nni_refine(D, tree, n, max_rounds=5):
    current_tree = tree
    current_score = _quartet_tree_score(D, current_tree, n)
    
    for _ in range(max_rounds):
        if _elapsed() > _DEADLINE:
            break
        improved = False
        best_new_tree = None
        best_new_score = current_score
        
        def _find_nni(node, parent=None):
            nonlocal improved, best_new_tree, best_new_score
            if isinstance(node, int):
                return
            left, right = node[0], node[1]
            if not isinstance(left, int) and not isinstance(right, int):
                a, b = left[0], left[1]
                c, d = right[0], right[1]
                s_cur = D[a,b] + D[c,d]
                s1 = D[a,c] + D[b,d]
                s2 = D[a,d] + D[b,c]
                
                if s1 < s_cur:
                    new_node = ((a, c), (b, d))
                    new_tree = _replace_subtree(current_tree, node, new_node)
                    if new_tree is not None:
                        new_score = current_score - s_cur + s1
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_tree = new_tree
                            improved = True
                
                if s2 < s_cur:
                    new_node = ((a, d), (b, c))
                    new_tree = _replace_subtree(current_tree, node, new_node)
                    if new_tree is not None:
                        new_score = current_score - s_cur + s2
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_tree = new_tree
                            improved = True
            
            _find_nni(left, node)
            _find_nni(right, node)
        
        _find_nni(current_tree)
        
        if improved:
            current_tree = best_new_tree
            current_score = best_new_score
        else:
            break
    
    return current_tree


def reconstruct(sequences):
    global _start_time
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    
    if _elapsed() > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    D = _jc_distance_matrix([sequences[lab] for lab in labels])
    
    if n <= 3:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]});"
    
    result_str = _nj_build(D)
    tree = _parse_newick(result_str)
    
    if _elapsed() < _DEADLINE - 10:
        tree = _nni_refine(D, tree, n)
    
    result_str = _to_newick(tree) + ";"
    
    def replace_labels(s):
        out = []
        i = 0
        while i < len(s):
            if s[i].isdigit():
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                idx = int(s[i:j])
                if idx in idx_to_label:
                    out.append(idx_to_label[idx])
                else:
                    out.append(s[i:j])
                i = j
            else:
                out.append(s[i])
                i += 1
        return "".join(out)
    
    result = replace_labels(result_str)
    
    if _elapsed() > _DEADLINE:
        return result
    
    return result
