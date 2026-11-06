# EVOLVE-BLOCK-START
"""Reconstruct unrooted topology: log-det/K2P distances + clean greedy
   3-addition (edge insertion) + NJ cross-check; per-instance best-of."""
import time
import math
import numpy as np

_DEADLINE = 255.0
_start_wall = time.time()
_CHAR_MAP = {65: 0, 67: 1, 71: 2, 84: 3}


def _now():
    return time.time() - _start_wall


def _encode(seq):
    out = np.empty(len(seq), dtype=np.uint8)
    for i, c in enumerate(seq):
        out[i] = _CHAR_MAP.get(ord(c), 0)
    return out


def _star(n):
    return "(" + ",".join("t%d" % i for i in range(n)) + ");"


def _distance_matrix(seqs, L):
    """log-det (K2P) pairwise distances; JC fallback per pair if singular."""
    n = seqs.shape[0]
    counts = np.zeros((4, L), dtype=np.int32)
    for a in range(4):
        counts[a] = (seqs == a).sum(axis=0)
    P = (counts @ counts.T) / float(4 * L)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            pi = P[i, i]; pj = P[j, j]; pij = P[i, j]
            if pi > 1e-9 and pj > 1e-9 and pij > 1e-9:
                val = 0.25 * math.log((pi * pj) / (pij * pij))
            else:
                p = pij / math.sqrt(pi * pj) if (pi * pj) > 0 else 0.0
                p = min(p, 0.7499)
                val = -0.75 * math.log(max(1.0 - 4.0 * p / 3.0, 1e-12))
            D[i, j] = D[j, i] = val
    np.fill_diagonal(D, 0.0)
    return D


def _nj(D):
    n = D.shape[0]
    if n <= 2:
        return "(" + ",".join("t%d" % i for i in range(n)) + ");"
    if n == 3:
        return "((t0,t1),t2);"
    D = D.astype(np.float64)
    names = list(range(n))
    while len(names) > 3:
        if _now() > _DEADLINE:
            break
        m = len(names)
        idx = np.arange(m)
        row = D[idx][:, idx]
        Q = (m - 2) * row - row.sum(axis=1)[:, None] - row.sum(axis=0)[None, :]
        i, j = np.unravel_index(np.argmin(Q), (m, m))
        if i == j:
            i, j = 0, 1
        a = names[i]; b = names[j]
        rest = [k for k in idx if k not in (i, j)]
        d_ab = D[i, j]
        d_new_r = 0.5 * (D[i, rest] + D[j, rest] - d_ab)
        D[i, rest] = d_new_r
        D[rest, i] = d_new_r
        names.remove(b)
    if len(names) == 3:
        a, b, c = names
        return "((t%d,t%d),t%d);" % (a, b, c)
    return "(" + ",".join("t%d" % x for x in names) + ");"


def _greedy(D):
    """Greedy 3-taxon addition with proper edge insertion. Returns Newick."""
    n = D.shape[0]
    if n <= 2:
        return "(" + ",".join("t%d" % i for i in range(n)) + ");"
    if n == 3:
        return "((t0,t1),t2);"
    D = D.astype(np.float64)
    # adjacency: undirected, node ids 0..n-1 leaves, n.. internal
    adj = {i: [] for i in range(n)}
    next_id = n

    def add_edge(u, v):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # seed: 3-leaf star with center u
    u = next_id; next_id += 1
    for x in (0, 1, 2):
        add_edge(u, x)

    in_tree = {0, 1, 2}

    def path_nodes(a, b):
        # BFS shortest path in current tree
        from collections import deque
        if a == b:
            return [a]
        prev = {a: None}
        q = deque([a])
        while q:
            cur = q.popleft()
            for nb in adj.get(cur, []):
                if nb not in prev:
                    prev[nb] = cur
                    if nb == b:
                        # reconstruct
                        path = [b]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(nb)
        return None

    for _ in range(n - 3):
        if _now() > _DEADLINE:
            break
        ext = [x for x in range(n) if x not in in_tree]
        if not ext:
            break
        # current nodes
        nodes = list(adj.keys())
        best = None
        best_strain = None
        for t in ext:
            for x in nodes:
                for y in nodes:
                    if x == y:
                        continue
                    strain = abs(D[t, x] + D[t, y] - D[x, y])
                    if best_strain is None or strain < best_strain:
                        best_strain = strain
                        best = (t, x, y)
        if best is None:
            break
        t, x, y = best
        # insert t by creating node z on the edge between x and y (split the path)
        path = path_nodes(x, y)
        if path is None or len(path) < 2:
            # attach t directly to x
            add_edge(x, t)
            in_tree.add(t)
            continue
        # pick the edge in the path closest to the barycenter of t
        # choose edge (p,q) minimizing |d(t,p)+d(t,q)-d(p,q)| already ~0; just use first edge
        p, q = path[0], path[1]
        z = next_id; next_id += 1
        # remove edge p-q
        adj[p].remove(q)
        adj[q].remove(p)
        # add z connected to p, q, t
        add_edge(z, p)
        add_edge(z, q)
        add_edge(z, t)
        in_tree.add(t)

    # serialize (unrooted) - pick any root
    def build(node, par):
        kids = [k for k in adj.get(node, []) if k != par]
        if node < n:
            lab = "t%d" % node
        else:
            lab = "n%d" % node
        if not kids:
            return lab
        return "(" + lab + "," + ",".join(build(k, node) for k in kids) + ")"
    root = 0
    return build(root, -1) + ";"


def _splits_from_newick(nw, n):
    def parse(s, pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                c, pos = parse(s, pos)
                children.append(c)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            return children, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',();':
                pos += 1
            return s[start:pos], pos
    try:
        root, _ = parse(nw, 0)
    except Exception:
        return set()

    def collect(node):
        if isinstance(node, list):
            all_l = set()
            child_sets = []
            for ch in node:
                sp, ls = collect(ch)
                all_l |= ls
                child_sets.append((sp, ls))
            splits = set()
            for sp_i, ls_i in child_sets:
                splits |= sp_i
                if 0 < len(ls_i) < len(all_l):
                    splits.add(frozenset(ls_i))
            return splits, all_l
        else:
            try:
                return set(), {int(node[1:])}
            except Exception:
                return set(), set()
    splits, _ = collect(root)
    return {s for s in splits if 1 < len(s) < n}


def _rf_norm(s1, s2):
    union = s1 | s2
    if not union:
        return 0.0
    return 1.0 - (2.0 * len(s1 & s2) / len(union))


def _tree_fit(D, nw):
    """Score a tree by how well its implied topology matches D via
       the number of quartets it gets 'right' (cheap proxy)."""
    n = D.shape[0]
    sp = _splits_from_newick(nw, n)
    if not sp:
        return 0.0
    # For each split, check consistency: the split should separate taxa
    # such that within-group mean distance < between-group mean distance.
    score = 0.0
    cnt = 0
    for s in sp:
        ls = set(s)
        if len(ls) < 1 or len(ls) >= n:
            continue
        other = set(range(n)) - ls
        within = 0.0; wn = 0
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                a = list(ls)[i]; b = list(ls)[j]
                within += D[a, b]; wn += 1
        between = 0.0; bn = 0
        for a in ls:
            for b in other:
                between += D[a, b]; bn += 1
        if wn and bn:
            # reward if within < between (consistent clustering)
            if within / wn < between / bn:
                score += 1.0
            cnt += 1
    return score / cnt if cnt else 0.0


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n <= 3:
        return _star(n)
    seqs = np.stack([_encode(sequences[l]) for l in labels])
    L = seqs.shape[1]
    if L == 0:
        return _star(n)

    D = _distance_matrix(seqs, L)
    if not np.any(D):
        return _star(n)

    cand = [_nj(D)]
    if _now() < _DEADLINE - 5:
        cand.append(_greedy(D))

    best = cand[0]
    best_fit = None
    for c in cand:
        f = _tree_fit(D, c)
        if best_fit is None or f > best_fit:
            best_fit = f
            best = c
    return best
# EVOLVE-BLOCK-END
