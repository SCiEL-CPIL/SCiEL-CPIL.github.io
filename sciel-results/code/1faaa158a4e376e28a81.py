"""Phylogenetic topology recovery: NJ start + Fitch-scored NNI refinement."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_BASE = np.array([1, 2, 4, 8], dtype=np.uint8)
_CHAR = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)


def _jc_distance(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.7499:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_masks(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    masks = np.zeros((L, n), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab].encode('ascii')
        idx = np.searchsorted(_CHAR, s)
        masks[:, i] = _BASE[idx]
    return masks


def _nj_start(masks):
    n = masks.shape[1]
    if n == 2:
        return (0, 1)
    if n == 3:
        return ((0, 1), 2)

    # vectorized pairwise JC distances
    diff = (masks[:, :, None] != masks[:, None, :]).sum(axis=0)
    p = diff / masks.shape[0]
    np.fill_diagonal(p, 0.0)
    safe = np.where(p < 0.7499, p, 0.0)
    D = -0.75 * np.log(1.0 - 4.0 * safe / 3.0)
    D = np.where(p < 0.7499, D, 10.0)

    active = list(range(n))
    cur = D.copy()
    counter = n
    while len(active) > 2:
        m = len(active)
        r = cur[active].sum(axis=1)
        best_q = float('inf')
        bi = bj = -1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * cur[active[i], active[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    bi, bj = i, j
        ai, aj = active[bi], active[bj]
        new = counter
        counter += 1
        rest = [x for x in active if x != ai and x != aj]
        row = 0.5 * (cur[ai] + cur[aj] - cur[ai, aj])
        row[ai] = row[aj] = row[new] = 0.0
        for x in rest:
            cur[x, new] = cur[new, x] = row[x]
        active = rest + [new]
    return (active[0], active[1])


def _tree_to_list(t):
    """Convert a tuple tree (leaves = int) into a flat node list.

    Returns (nodes, root_idx, leaf_node). nodes is a list of [a, b] where
    a, b are child indices for internal nodes, or -1 for leaves.
    """
    nodes = []
    leaf_node = {}

    def rec(t):
        if isinstance(t, int):
            idx = len(nodes)
            nodes.append([-1, -1])
            leaf_node[t] = idx
            return idx
        a = rec(t[0])
        b = rec(t[1])
        idx = len(nodes)
        nodes.append([a, b])
        return idx

    root = rec(t)
    return nodes, root, leaf_node


def _list_to_tree(nodes, idx):
    a, b = nodes[idx]
    if a == -1:
        return idx
    return (_list_to_tree(nodes, a), _list_to_tree(nodes, b))


def _newick(t):
    if isinstance(t, int):
        return "t%d" % t
    return "(%s,%s)" % (_newick(t[0]), _newick(t[1]))


def _compute_masks(nodes, leaf_node, seq_masks):
    n = len(nodes)
    nm = np.empty((n, seq_masks.shape[0]), dtype=np.uint8)
    for leaf, node in leaf_node.items():
        nm[node] = seq_masks[:, leaf]
    for idx in range(n - 1, -1, -1):
        a, b = nodes[idx]
        if a != -1:
            nm[idx] = nm[a] | nm[b]
    return nm


def _fitch(nm, nodes, idx):
    a, b = nodes[idx]
    if a == -1:
        return 0
    la = nm[a]
    lb = nm[b]
    return _fitch(nm, nodes, a) + _fitch(nm, nodes, b) + int(np.count_nonzero(la & lb == 0))


def _get_internal_edges(nodes, root):
    edges = []
    for idx in range(root + 1):
        a, b = nodes[idx]
        if a != -1:
            edges.append((idx, a))
            edges.append((idx, b))
    return edges


def _nni_score(nm, nodes, u, v, w, x, y):
    # original: u=(v,w), v=(x,y)
    ux = nm[u] & nm[x]
    uy = nm[u] & nm[y]
    wx = nm[w] & nm[x]
    wy = nm[w] & nm[y]
    uy2 = nm[u] & nm[y]
    # arrangement A: u=(x, v'), v'=(w,y)  -> (u,x) & (u,y) & (w,x)
    sA = int(np.count_nonzero(ux == 0)) + int(np.count_nonzero(uy == 0)) + int(np.count_nonzero(wx == 0))
    # arrangement B: u=(y, v'), v'=(w,x)  -> (u,y) & (u,x) & (w,y)
    sB = int(np.count_nonzero(uy2 == 0)) + int(np.count_nonzero(ux == 0)) + int(np.count_nonzero(wy == 0))
    return min(sA, sB)


def _apply_nni(nodes, u, v, w, x, y, kind):
    if kind == 1:
        nodes[u] = [x, v]
        nodes[v] = [w, y]
    else:
        nodes[u] = [y, v]
        nodes[v] = [w, x]


def _revert_nni(nodes, u, v, w, x, y):
    nodes[u] = [v, w]
    nodes[v] = [x, y]


def _nni_polish(nodes, root, nm, seq_masks, deadline):
    n = len(nodes)
    L = seq_masks.shape[0]
    best = _fitch(nm, nodes, root)

    while True:
        improved = False
        for u, v in _get_internal_edges(nodes, root):
            if time.time() > deadline:
                return best
            a, b = nodes[v]
            if a == -1 or b == -1:
                continue
            if u == v:
                continue
            if nodes[u][0] == v:
                w = nodes[u][1]
            else:
                w = nodes[u][0]
            x, y = a, b
            cand = _nni_score(nm, nodes, u, v, w, x, y)
            if cand < best:
                _apply_nni(nodes, u, v, w, x, y, 1)
                nm[v] = nm[w] | nm[y]
                nm[u] = nm[x] | nm[v]
                best = cand
                improved = True
            elif cand == best:
                # tie: try the other quartet arrangement to break symmetry
                ux = nm[u] & nm[x]
                uy = nm[u] & nm[y]
                wy = nm[w] & nm[y]
                sB = int(np.count_nonzero(uy == 0)) + int(np.count_nonzero(ux == 0)) + int(np.count_nonzero(wy == 0))
                if sB < best:
                    _apply_nni(nodes, u, v, w, x, y, 2)
                    nm[v] = nm[w] | nm[x]
                    nm[u] = nm[y] | nm[v]
                    best = sB
                    improved = True
        if not improved or time.time() > deadline:
            break
    return best


def reconstruct(sequences):
    start = time.time()
    deadline = start + 255.0  # safety margin before the 270s wall-clock kill

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(%s,%s);" % (labels[0], labels[1])

    seq_masks = _build_masks(sequences, labels)
    start_tree = _nj_start(seq_masks)

    nodes, root, leaf_node = _tree_to_list(start_tree)
    nm = _compute_masks(nodes, leaf_node, seq_masks)
    _nni_polish(nodes, root, nm, seq_masks, deadline)

    return _newick(_list_to_tree(nodes, root)) + ";"
# EVOLVE-BLOCK-END
