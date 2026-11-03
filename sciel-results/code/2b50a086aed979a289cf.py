"""Greedy + vectorized local search for K_3(7,1); SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE_FRACTION = 0.95
_DEADLINE = 145.0


def construct(q, n, radius):
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 145.0
    if q != 3 or n != 7 or radius != 1:
        return _fallback(q, n, radius, t0)

    N = 3 ** n
    # Precompute every word's radius-1 neighbourhood (N x (1 + 2n) int16).
    neigh = np.empty((N, 1 + 2 * n), dtype=np.int16)
    neigh[:, 0] = np.arange(N, dtype=np.int16)
    base = np.array([3 ** i for i in range(n)], dtype=np.int16)
    for i in range(n):
        w = np.arange(N, dtype=np.int16)
        r = (w // base[i]) % 3
        neigh[:, 1 + 2 * i] = w - base[i] * (r == 0) + base[i] * (r == 2)
        neigh[:, 2 + 2 * i] = w + base[i] * (r == 2) - base[i] * (r == 0)

    # Ball masks: mask[c][w] = 1 iff w is within radius of codeword c.
    masks = np.zeros((N, N), dtype=np.int8)
    masks[np.arange(N)[:, None], neigh] = 1

    def is_covered(idx, covered):
        return masks[idx].any(axis=0)

    def prune(idx, covered):
        idx = list(idx)
        while True:
            m = masks[idx]
            cnt = m.sum(axis=0)
            keep = np.ones(len(idx), dtype=bool)
            for c in range(len(idx)):
                if (cnt == 1).any():
                    cnt -= m[c]
                else:
                    keep[c] = False
            if keep.all():
                break
            idx = [idx[c] for c in range(len(idx)) if keep[c]]
            covered = masks[idx].any(axis=0)
        return idx, covered

    def local_search(idx, covered):
        idx = list(idx)
        covered = np.asarray(covered, dtype=bool)
        while time.time() < _DEADLINE:
            changed = False
            # ---- add moves (covers >= 2 uncovered) ----
            unc = ~covered
            if unc.any():
                u = np.where(unc)[0]
                g = masks[idx][:, u].sum(axis=1)
                best = int(np.argmax(g))
                if g[best] >= 2:
                    idx.append(idx[best])
                    covered |= masks[idx[best]]
                    changed = True
            # ---- replace moves (swap redundant codeword for word covering >= 3) ----
            if not changed and len(idx) > 1:
                m = masks[idx]
                cnt = m.sum(axis=0)
                redundant = np.array([(cnt > 1).all() for c in range(len(idx))])
                if redundant.any():
                    unc = ~covered
                    if unc.any():
                        u = np.where(unc)[0]
                        g = masks[:, u].sum(axis=1)
                        bi = int(np.argmax(g))
                        if g[bi] >= 3:
                            ri = int(np.argmax(redundant))
                            idx[ri] = bi
                            covered = masks[idx].any(axis=0)
                            changed = True
            if not changed:
                break
        # final pruning
        idx, covered = prune(idx, covered)
        return idx, covered

    def greedy_init(seed):
        rng = np.random.default_rng(seed)
        order = list(range(N))
        rng.shuffle(order)
        idx = []
        covered = np.zeros(N, dtype=bool)
        for c in order:
            if time.time() > _DEADLINE:
                break
            if not covered[c]:
                idx.append(c)
                covered |= masks[c]
            if covered.all():
                break
        return idx, covered

    best = None
    best_len = N + 1
    seed = 0
    while time.time() < _DEADLINE:
        seed += 1
        idx, covered = greedy_init(seed)
        idx, covered = local_search(idx, covered)
        if len(idx) < best_len:
            best = idx
            best_len = len(idx)
            if best_len <= 186:
                break
    if best is None:
        best = list(range(N))
    return [list(np.array_divmod(w, [3 ** i for i in range(n - 1, -1, -1)]).tolist()) for w in best]


def _fallback(q, n, radius, t0):
    # Generic fallback: greedy set cover + simple prune for non-(3,7,1) calls.
    import numpy as np
    N = q ** n
    words = [list(np.array_divmod(w, [q ** i for i in range(n - 1, -1, -1)])) for w in range(N)]
    # neighbourhood radius `radius`
    def neigh_of(w):
        out = [w]
        for i in range(n):
            for d in range(-radius, radius + 1):
                if d == 0:
                    continue
                v = w[i] + d
                if 0 <= v < q:
                    x = list(w)
                    x[i] = v
                    out.append(int(np.array(x) @ np.array([q ** (n - 1 - j) for j in range(n)])))
        return out
    masks = np.zeros((N, N), dtype=np.int8)
    for w in range(N):
        for v in neigh_of(w):
            masks[w, v] = 1
    covered = np.zeros(N, dtype=bool)
    idx = []
    while not covered.all() and time.time() < t0 + 140:
        gains = masks[:, ~covered].sum(axis=1) if (~covered).any() else np.zeros(N)
        c = int(np.argmax(gains))
        if gains[c] <= 0:
            break
        idx.append(c)
        covered |= masks[c]
    # prune
    idx = list(idx)
    while True:
        m = masks[idx]
        cnt = m.sum(axis=0)
        keep = np.ones(len(idx), dtype=bool)
        for c in range(len(idx)):
            if (cnt == 1).any():
                cnt -= m[c]
            else:
                keep[c] = False
        if keep.all():
            break
        idx = [idx[c] for c in range(len(idx)) if keep[c]]
    return [words[w] for w in idx]
# EVOLVE-BLOCK-END
