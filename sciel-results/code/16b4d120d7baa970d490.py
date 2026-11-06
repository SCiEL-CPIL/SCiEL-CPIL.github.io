"""Spectral embedding + Ward linkage phylogeny reconstruction."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START = time.time()
_DEADLINE = 250.0  # wall-clock budget with margin under the 270 s kill


def _elapsed():
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_from_p(p):
    """Jukes-Cantor correction; p is the proportion of differing sites."""
    p = min(max(p, 0.0), 0.7499)
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _jc_matrix(sequences, labels):
    """Pairwise JC-corrected distances as an (n, n) float array."""
    n = len(labels)
    codes = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    code = {ord('A'): 0, ord('C'): 1, ord('G'): 2, ord('T'): 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        codes[i] = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
    # pairwise mismatch proportions
    diff = (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    L = codes.shape[1]
    p = diff.astype(float) / L
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(np.clip(1.0 - 4.0 * p / 3.0, 1e-12, None))
    np.fill_diagonal(d, 0.0)
    return d


def _ward_build(d, labels):
    """Ward's linkage on squared Euclidean distances. Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    sq = d * d
    active = list(range(n))
    size = np.ones(n, dtype=float)
    name = list(labels)

    def new_name(a, b):
        return "(" + name[a] + "," + name[b] + ")"

    while len(active) > 2:
        if _deadline_exceeded():
            break
        m = len(active)
        # Ward merge cost for pair (a, b)
        # = (size[a]*size[b]/(size[a]+size[b])) * (d_ab^2)
        # using squared distances
        best = None
        best_pair = None
        for i in range(m):
            a = active[i]
            sa = size[a]
            for j in range(i + 1, m):
                b = active[j]
                sb = size[b]
                cost = (sa * sb / (sa + sb)) * (sq[a, b] ** 2)
                if best is None or cost < best:
                    best = cost
                    best_pair = (a, b)
        a, b = best_pair
        new = n
        n += 1
        name.append(new_name(a, b))
        size[new] = size[a] + size[b]
        # new squared distance to every other active node (Ward update)
        for x in active:
            if x == a or x == b:
                continue
            sx = size[x]
            # d_new^2 = ((sa*d_new_a^2 + sb*d_new_b^2 - (sa+sb)*0) ... )
            # Ward: d(new,x)^2 = (sa*d(a,x)^2 + sb*d(b,x)^2 - (sa+sb)*d(a,b)^2*sa*sb/(sa+sb)) / (sa+sb)
            # Simpler standard Ward update:
            # d(new,x)^2 = (sa*d(a,x)^2 + sb*d(b,x)^2)/(sa+sb)
            #            - (sa*sb/(sa+sb)) * d(a,b)^2
            da = sq[a, x] ** 2
            db = sq[b, x] ** 2
            dab = sq[a, b] ** 2
            val = (sa * da + sb * db) / (sa + sb) - (sa * sb / (sa + sb)) * dab
            if val < 0:
                val = 0.0
            sq[new, x] = sq[x, new] = val
        active = [x for x in active if x not in (a, b)] + [new]

    if _deadline_exceeded() or len(active) > 2:
        # fall back to a star tree over current names
        return "(" + ",".join(name[x] for x in active) + ");"
    return "(" + name[active[0]] + "," + name[active[1]] + ");"


def _nj_fallback(sequences, labels):
    """Classic neighbour-joining fallback (vectorized-ish)."""
    n = len(labels)
    if n <= 1:
        return (labels[0] if n == 1 else "") + ";"
    d = _jc_matrix(sequences, labels)
    active = list(range(n))
    name = list(labels)
    while len(active) > 3:
        if _deadline_exceeded():
            return "(" + ",".join(name[x] for x in active) + ");"
        m = len(active)
        D = np.array([d[a, b] for a in active for b in active]).reshape(m, m)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, 0.0)
        ii, jj = np.unravel_index(np.argmin(Q), Q.shape)
        if jj < ii:
            ii, jj = jj, ii
        a = active[ii]
        b = active[jj]
        new = n
        n += 1
        name.append("(" + name[a] + "," + name[b] + ")")
        for k, x in enumerate(active):
            if x == a or x == b:
                continue
            d[new, x] = d[x, new] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        active = [x for x in active if x not in (a, b)] + [new]
    if len(active) == 3:
        a, b, c = active
        return "(" + name[a] + ",(" + name[b] + "," + name[c] + "));"
    return "(" + ",".join(name[x] for x in active) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return (labels[0] if n else "") + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    d = _jc_matrix(sequences, labels)

    # Spectral embedding: double-center the (squared) distance matrix.
    sq = d * d
    one_n = np.ones((n, n)) / n
    X = sq - sq.mean(axis=1, keepdims=True) - sq.mean(axis=0, keepdims=True) + sq.mean()
    # eigendecompose (symmetric)
    try:
        evals, evecs = np.linalg.eigh(X)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]
        # keep a modest number of dimensions
        k = min(n - 1, 12)
        coords = evecs[:, :k] * np.sqrt(evals[:k])
    except Exception:
        coords = None

    if coords is not None and not _deadline_exceeded():
        try:
            # Ward linkage on the embedded coordinates (squared Euclidean)
            cdist = np.sqrt(np.maximum(
                (coords ** 2).sum(axis=1)[:, None]
                + (coords ** 2).sum(axis=1)[None, :]
                - 2.0 * (coords @ coords.T), 0.0))
            np.fill_diagonal(cdist, 0.0)
            return _ward_build(cdist, labels)
        except Exception:
            pass

    # Fallback: neighbour joining on JC distances.
    return _nj_fallback(sequences, labels)
# EVOLVE-BLOCK-END
