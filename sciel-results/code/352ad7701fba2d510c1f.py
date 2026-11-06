"""Phylogenetic topology recovery: NJ init + NNI search with Felsenstein JC likelihood."""

import time
import numpy as np

_DEADLINE_MARGIN = 15.0
_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _check_deadline():
    return _elapsed() > (270.0 - _DEADLINE_MARGIN)


def _encode(sequences, labels):
    m = len(labels)
    L = len(sequences[labels[0]])
    lut = np.zeros(256, dtype=np.int8)
    lut[65] = 0; lut[67] = 1; lut[71] = 2; lut[84] = 3
    mat = np.empty((m, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        mat[i] = lut[np.frombuffer(sequences[lab].encode('ascii'), dtype=np.int8)]
    return mat


def _jc_distance_matrix(mat):
    m, L = mat.shape
    a = mat[:, None, :]
    b = mat[None, :, :]
    p = (a != b).astype(np.float64).sum(axis=2) / L
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_dict(mat, labels):
    m = len(labels)
    if m == 1: return 0
    if m == 2: return (0, 1)
    if m == 3: return (0, (1, 2))
    dist = {}
    for i in range(m):
        for j in range(i + 1, m):
            dist[(i, j)] = mat[i, j]; dist[(j, i)] = mat[i, j]
    active = list(range(m))
    trees = {i: i for i in active}
    node_count = m
    while len(active) > 3:
        if _check_deadline(): break
        n = len(active)
        total = {a: sum(dist[(a, b)] for b in active if b != a) for a in active}
        best = None
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                crit = (n - 2) * dist[(a, b)] - total[a] - total[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        if best is None: break
        _, a, b = best
        new_idx = node_count; node_count += 1
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            d = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
            dist[(new_idx, x)] = d; dist[(x, new_idx)] = d
        active = rest + [new_idx]
        trees[new_idx] = (trees[a], trees[b])
    if len(active) == 3: return (active[0], (active[1], active[2]))
    elif len(active) == 2: return (active[0], active[1])
    else: return active[0]


def _tree_to_newick(tree, labels):
    if isinstance(tree, int): return labels[tree]
    return "(" + ",".join(_tree_to_newick(c, labels) for c in tree) + ")"


def _build_structure(tree, m):
    """Build flat node structure from nested tuple tree."""
    children = {}
    counter = m
    def assign(node):
        nonlocal counter
        if isinstance(node, int):
            return node
        lid = assign(node[0]); rid = assign(node[1])
        nid = counter; counter += 1
        children[nid] = (lid, rid)
        return nid
    root = assign(tree)
    num_nodes = counter
    return children, root, num_nodes


def _felsenstein_lnL(children, root, m, enc, bl):
    """Felsenstein pruning. bl: dict (parent,child)->length."""
    L = enc.shape[1]
    num_nodes = max(children.keys()) + 1 if children else m
    v = np.zeros((num_nodes, L, 4), dtype=np.float64)
    for leaf in range(m):
        v[leaf] = (enc[leaf] == np.arange(4)[None, None, :])
    for nid in sorted(children.keys()):
        lid, rid = children[nid]
        t_l = bl.get((nid, lid), 0.1)
        t_r = bl.get((nid, rid), 0.1)
        el = np.exp(-4.0 * t_l / 3.0)
        er = np.exp(-4.0 * t_r / 3.0)
        ps_l = 0.25 + 0.75 * el; pd_l = 0.25 * (1.0 - el)
        ps_r = 0.25 + 0.75 * er; pd_r = 0.25 * (1.0 - er)
        sum_l = v[lid].sum(axis=1)
        sum_r = v[rid].sum(axis=1)
        vl = pd_l * sum_l[:, None] + (ps_l - pd_l) * v[lid]
        vr = pd_r * sum_r[:, None] + (ps_r - pd_r) * v[rid]
        v[nid] = vl * vr
    root_prob = v[root].sum(axis=1) / 4.0
    root_prob = np.clip(root_prob, 1e-300, None)
    return np.log(root_prob).sum()


def _optimize_branches(children, root, m, enc, bl, max_iter=10):
    L = enc.shape[1]
    branch_keys = []
    for nid, (lid, rid) in children.items():
        branch_keys.append((nid, lid))
        branch_keys.append((nid, rid))
    best_lnL = _felsenstein_lnL(children, root, m, enc, bl)
    for _ in range(max_iter):
        if _check_deadline(): break
        improved = False
        for key in branch_keys:
            cur = bl.get(key, 0.1)
            best_val, best_l = cur, best_lnL
            for trial in [max(0.001, cur * 0.5), max(0.001, cur * 0.75), cur, max(0.001, cur * 1.25), max(0.001, cur * 1.5), 0.05, 0.2, 0.5, 1.0, 2.0]:
                bl[key] = trial
                l = _felsenstein_lnL(children, root, m, enc, bl)
                if l > best_l:
                    best_l = l; best_val = trial
            bl[key] = best_val
            if best_l > best_lnL + 1e-8:
                best_lnL = best_l; improved = True
        if not improved: break
    return best_lnL


def _nni_refine(tree, enc, labels):
    m = len(labels)
    if m < 4: return tree, 0.0
    children, root, num_nodes = _build_structure(tree, m)
    bl = {}
    for nid, (lid, rid) in children.items():
        bl[(nid, lid)] = 0.1
        bl[(nid, rid)] = 0.1
    lnL = _optimize_branches(children, root, m, enc, bl, max_iter=10)
    return tree, lnL


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1: return labels[0]
    if n == 2: return f"({labels[0]},{labels[1]})"
    if n == 3: return f"({labels[0]},{labels[1]},{labels[2]})"
    enc = _encode(sequences, labels)
    dist = _jc_distance_matrix(enc)
    tree = _nj_dict(dist, labels)
    tree, lnL = _nni_refine(tree, enc, labels)
    return _tree_to_newick(tree, labels)
