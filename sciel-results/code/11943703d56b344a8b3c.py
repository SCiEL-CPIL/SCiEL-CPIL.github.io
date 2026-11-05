"""Phylogenetic topology recovery: fast vectorized NJ + time-guarded split refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = time.time() + 245.0          # hard wall-clock safety margin (< 270 s)


def _time_left():
    return _DEADLINE - time.time()


def _jc_matrix(seqmat, L):
    """JC-corrected pairwise distances from a 0/1/2/3 integer (n, L) matrix."""
    n = seqmat.shape[0]
    d = np.zeros((n, n))
    for j in range(L):
        col = seqmat[:, j]
        D = (col[:, None] != col[None, :]).astype(np.float64)
        np.fill_diagonal(D, 0.0)
        d += D
    p = d / L
    np.clip(p, 0.0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_newick(D, labels):
    """Vectorized Neighbor Joining. D: (n,n) float, labels: list of str. Returns Newick."""
    n = D.shape[0]
    if n < 3:
        return labels[0] + ";" if n == 1 else "(" + ",".join(labels) + ");"
    D = D.astype(np.float64, copy=True)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    while len(nodes) > 3:
        m = len(nodes)
        idx = np.array(nodes)
        sub = D[np.ix_(idx, idx)]
        r = sub.sum(axis=1)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        ii, jj = np.unravel_index(np.argmin(Q), Q.shape)
        a, b = idx[ii], idx[jj]
        rest = [x for x in nodes if x not in (a, b)]
        new = n
        n += 1
        trees[new] = "(" + trees[a] + "," + trees[b] + ")"
        if rest:
            rix = np.array(rest)
            D[new, rix] = 0.5 * (D[a, rix] + D[b, rix] - D[a, b])
            D[rix, new] = D[new, rix]
        nodes = rest + [new]
    i, j, k = nodes
    return "(" + trees[i] + "," + trees[j] + "," + trees[k] + ");"


def _fitch(seqmat, L):
    """Fast Fitch parsimony. Returns (score, tree) with tree[i][j] = 0/1/2/3/4 (4=any)."""
    n = seqmat.shape[0]
    tree = np.zeros((n, n, L), dtype=np.int8)
    for j in range(L):
        col = seqmat[:, j]
        for i in range(n):
            for k in range(i + 1, n):
                tree[i, k, j] = col[i] if col[i] == col[k] else 4
    score = 0
    for j in range(L):
        T = tree[:, :, j]
        for i in range(n):
            for k in range(i + 1, n):
                best = None
                for m in range(n):
                    if m == i or m == k:
                        continue
                    if T[i, m] == 4 or T[m, k] == 4:
                        c = 4
                    elif T[i, m] == T[m, k]:
                        c = T[i, m]
                    else:
                        c = 4
                    if c != 4:
                        if best is None or c < best:
                            best = c
                    elif best is None:
                        best = 4
                if best == 4:
                    score += 1
                tree[i, k, j] = best
    return score, tree


def _splits_from_tree(tree, n):
    """Set of frozensets (splits) from a binary-compatible distance-like tree."""
    splits = set()
    for i in range(n):
        for k in range(i + 1, n):
            S = frozenset(m for m in range(n) if m != i and m != k and tree[i, k, m] != 4)
            if len(S) >= 2 and len(S) <= n - 2:
                splits.add(S)
    return splits


def _stability(tree, n, seqmat, L, sample=200):
    """Heuristic score: how consistent the inferred splits are with the data."""
    if tree is None:
        return float("inf")
    splits = _splits_from_tree(tree, n)
    if not splits:
        return float("inf")
    rng = np.random.default_rng(12345)
    idx = rng.choice(L, size=min(sample, L), replace=False)
    total = 0
    cnt = 0
    for j in idx:
        col = seqmat[:, j]
        for S in splits:
            a = list(S)
            b = [m for m in range(n) if m not in S]
            va = np.bincount(col[a], minlength=4).sum()
            vb = np.bincount(col[b], minlength=4).sum()
            total += min(va, vb)
            cnt += 1
    return total / cnt if cnt else float("inf")


def _tree_to_newick(tree, n, labels):
    """Build a Newick from a binary 'tree' matrix (0/1/2/3/4) via NJ on a proxy matrix."""
    D = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            D[i, k] = D[k, i] = 1.0 if tree[i, k, 0] == 4 else 0.0
    return _nj_newick(D, labels)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    seqmat = np.array([[int(c) for c in s] for s in (sequences[lab] for lab in labels)],
                      dtype=np.int8)

    # Baseline: fast vectorized NJ
    D = _jc_matrix(seqmat, L)
    nj_newick = _nj_newick(D, labels)
    best_newick = nj_newick
    best_score = _stability(D, n, seqmat, L)

    if n <= 3:
        return "(" + ",".join(labels) + ");"

    # Time-guarded refinement: Fitch parsimony + NJ-from-parsimony, keep best by stability
    while _time_left() > 1.0:
        _fitch_score, fitch_tree = _fitch(seqmat, L)
        p_newick = _tree_to_newick(fitch_tree, n, labels)
        p_score = _stability(fitch_tree, n, seqmat, L)
        if p_score < best_score:
            best_score = p_score
            best_newick = p_newick
        if _time_left() < 1.0:
            break
        # one refinement pass on the best tree's proxy matrix, then stop
        D2 = np.zeros((n, n))
        for i in range(n):
            for k in range(i + 1, n):
                D2[i, k] = D2[k, i] = 1.0 if fitch_tree[i, k, 0] == 4 else 0.0
        r_newick = _nj_newick(D2, labels)
        r_score = _stability(D2, n, seqmat, L)
        if r_score < best_score:
            best_score = r_score
            best_newick = r_newick
        break  # refinement is a single pass; avoid unbounded loops

    return best_newick if best_newick else nj_newick
# EVOLVE-BLOCK-END
