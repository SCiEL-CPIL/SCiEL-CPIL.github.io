"""Phylogenetic topology recovery: NJ + bootstrap consensus (majority-rule)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


_START = time.perf_counter()
_BUDGET = 255.0  # safety margin below the 270s wall-clock kill


def _time_left():
    return _BUDGET - (time.perf_counter() - _START)


_INT_MAP = np.empty(256, dtype=np.int8)
_INT_MAP[ord('A')] = 0
_INT_MAP[ord('C')] = 1
_INT_MAP[ord('G')] = 2
_INT_MAP[ord('T')] = 3


def _seq_int(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    data = np.empty((n, L), dtype=np.int8)
    for i, lb in enumerate(labels):
        s = sequences[lb]
        data[i] = _INT_MAP[[ord(c) for c in s]]
    return data


def _dist_from_cols(cols, L):
    n = cols.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    chunk = 500
    for s0 in range(0, L, chunk):
        s1 = min(s0 + chunk, L)
        block = cols[:, s0:s1]
        diff = (block[:, None, :] != block[None, :, :])
        D += diff.sum(axis=2)
    p = D / L
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return D


def _nj_newick(D, labels):
    n = len(labels)
    D = D.astype(np.float64, copy=True)
    active = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    while len(active) > 3:
        m = len(active)
        r = np.array(active)
        sub = D[np.ix_(r, r)]
        row = sub.sum(axis=1)
        Q = (m - 2.0) * sub - row[:, None] - row[None, :]
        np.fill_diagonal(Q, 1e18)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        a = int(r[i]); b = int(r[j])
        new_id = -1 - len(active)
        tree[new_id] = "({},{})".format(tree[a], tree[b])
        rest = [x for x in active if x != a and x != b]
        for x in rest:
            D[new_id, x] = D[x, new_id] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        active = rest + [new_id]
    return "(" + ",".join(tree[x] for x in active) + ");"


def _splits(newick, labels):
    idx = {lb: i for i, lb in enumerate(labels)}
    n = len(labels)

    def split_top(s):
        parts, depth, cur = [], 0, ''
        for c in s:
            if c == '(':
                depth += 1; cur += c
            elif c == ')':
                depth -= 1; cur += c
            elif c == ',' and depth == 0:
                parts.append(cur); cur = ''
            else:
                cur += c
        if cur:
            parts.append(cur)
        return parts

    def walk(s):
        s = s.strip()
        if s.startswith('('):
            depth, end = 0, -1
            for k in range(len(s)):
                if s[k] == '(':
                    depth += 1
                elif s[k] == ')':
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
            acc = set()
            for p in split_top(s[1:end]):
                acc |= walk(p)
            return acc
        return {idx[s]}

    s = newick.strip().rstrip(';').strip()
    out = set()
    for p in split_top(s):
        A = walk(p)
        if 0 < len(A) < n:
            out.add(frozenset(A))
    return out


def _tree_from_splits(splits, labels):
    n = len(labels)
    if len(splits) != 2 * n - 3:
        return None
    leaf_set = frozenset(range(n))
    sp_list = list(splits)

    def find_split(leaves):
        for sp in sp_list:
            A = sp & leaves
            if 0 < len(A) < len(leaves):
                return A, leaves - A
        return None

    def emit(leaves):
        if len(leaves) == 1:
            return labels[next(iter(leaves))]
        part = find_split(leaves)
        if part is None:
            return None
        A, B = part
        sa = emit(A)
        sb = emit(B)
        if sa is None or sb is None:
            return None
        return "({},{})".format(sa, sb)

    res = emit(leaf_set)
    return ("(" + res + ");") if res is not None else None


def reconstruct(sequences):
    global _START
    _START = time.perf_counter()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"

    data = _seq_int(sequences, labels)
    L = data.shape[1]
    D = _dist_from_cols(data, L)
    base_newick = _nj_newick(D, labels)

    if n < 6 or L < 60 or _time_left() < 20.0:
        return base_newick

    rng = np.random.default_rng(20240601)
    site_idx = np.arange(L)
    counts = {}
    B = 60
    for b in range(B):
        if _time_left() < 20.0:
            break
        sample = rng.choice(site_idx, size=L, replace=True)
        cols = data[:, sample]
        Db = _dist_from_cols(cols, L)
        nw = _nj_newick(Db, labels)
        for f in _splits(nw, labels):
            counts[f] = counts.get(f, 0) + 1

    if not counts:
        return base_newick

    thr = 0.5 * B
    chosen = [f for f, c in counts.items() if c >= thr]
    cons = _tree_from_splits(set(chosen), labels)
    if cons is not None:
        return cons
    return base_newick
# EVOLVE-BLOCK-END
