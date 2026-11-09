"""Reconstruct unrooted topology from JC-simulated sequences.

Uses BIONJ (Gascuel 1997) with vectorized JC distance computation.
"""
import time

import numpy as np

_DEADLINE = time.time() + 250.0  # safety margin below the 270 s wall-clock kill


def _time_ok():
    return time.time() < _DEADLINE


def _jc_matrix(sequences, labels):
    """Vectorized Jukes-Cantor corrected pairwise distances."""
    enc = np.empty((len(labels), len(sequences[labels[0]])), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        enc[i] = (s == 'A').astype(np.int8) + (s == 'C').astype(np.int8) * 2 \
            + (s == 'G').astype(np.int8) * 4 + (s == 'T').astype(np.int8) * 8
    L = enc.shape[1]
    # Hamming distance matrix via 4 one-hot channels
    A = (enc == 1).astype(np.int16)
    C = (enc == 2).astype(np.int16)
    G = (enc == 4).astype(np.int16)
    T = (enc == 8).astype(np.int16)
    same = A @ A.T + C @ C.T + G @ G.T + T @ T.T
    dist = L - same
    np.fill_diagonal(dist, 0)
    p = dist.astype(np.float64) / L
    # JC correction: -0.75 * ln(1 - 4p/3), with a safe cap
    arg = 1.0 - 4.0 * p / 3.0
    np.clip(arg, 1e-9, None, out=arg)
    D = -0.75 * np.log(arg)
    D[~np.isfinite(D)] = 3.0
    return D


def _bionj(D, labels):
    """BIONJ reconstruction. Returns a Newick string (no branch lengths)."""
    n = len(labels)
    if n == 2:
        return "(%s,%s);" % (labels[0], labels[1])
    if n == 3:
        return "((%s,%s),%s);" % (labels[0], labels[1], labels[2])

    D = D.astype(np.float64, copy=True)
    trees = {i: labels[i] for i in range(n)}
    alive = list(range(n))

    while len(alive) > 3:
        if not _time_ok():
            break
        m = len(alive)
        r = np.array(alive, dtype=np.int64)
        sub = D[np.ix_(r, r)]
        rowsum = sub.sum(axis=1)

        # BIONJ beta correction
        beta = (rowsum - sub.mean(axis=1) * m) / (m - 2)
        # Q criterion matrix (symmetric)
        Q = (m - 2) * sub - beta[:, None] - beta[None, :]
        np.fill_diagonal(Q, np.inf)

        # pick the minimum off-diagonal pair
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = int(idx[0]), int(idx[1])
        if i == j:
            break
        a, b = r[i], r[j]

        # BIONJ branch lengths
        d_ab = D[a, b]
        da = rowsum[i]
        db = rowsum[j]
        beta_a = beta[i]
        beta_b = beta[j]
        la = 0.5 * d_ab + (da - db - beta_a + beta_b) / (2.0 * (m - 2))
        lb = 0.5 * d_ab - (da - db - beta_a + beta_b) / (2.0 * (m - 2))
        la = max(la, 0.0)
        lb = max(lb, 0.0)

        new_idx = len(alive)  # index into the full D array for the new node
        joined = "(%s,%s);" % (trees[a], trees[b])
        trees[new_idx] = joined

        # BIONJ distance update for the new node vs every other alive node
        rest = [x for x in r if x != a and x != b]
        D[new_idx, new_idx] = 0.0
        for k, x in enumerate(rest):
            val = 0.5 * (D[a, x] + D[b, x] - d_ab) + 0.5 * (la + lb)
            D[new_idx, x] = val
            D[x, new_idx] = val

        alive = rest + [new_idx]

    if len(alive) == 3:
        return "((%s,%s),%s);" % (trees[alive[0]], trees[alive[1]], trees[alive[2]])
    # fallback: star over whatever remains
    return "(" + ",".join(trees[x] for x in alive) + ");"


def _star(labels):
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return "(" + ",".join(labels) + ");"
    try:
        D = _jc_matrix(sequences, labels)
        return _bionj(D, labels)
    except Exception:
        return _star(labels)
