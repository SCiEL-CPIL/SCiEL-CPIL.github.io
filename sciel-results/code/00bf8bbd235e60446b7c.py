import time
import random
import numpy as np

_T0 = time.time()
_DEADLINE = 85.0          # hard safety margin vs 90s wall-clock
_N = 30
_ALL = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
_ALL_IDX = {tuple(p): i for i, p in enumerate(_ALL)}
_NALL = len(_ALL)
_ALL_F = _ALL.astype(np.float64)
_PAIR_AREA = np.empty((_NALL, _NALL), dtype=np.int64)
for i in range(_NALL):
    xi, yi = _ALL[i]
    for j in range(_NALL):
        xj, yj = _ALL[j]
        _PAIR_AREA[i, j] = abs(int(xi) * int(yj) - int(xj) * int(yi))
# vector difference counts for fast scoring
_DIFF_CNT = np.zeros((_NALL, _NALL), dtype=np.int64)
for i in range(_NALL):
    diffs = _ALL - _ALL[i]
    _DIFF_CNT[i] = np.bincount(diffs.ravel(), minlength=2 * _NALL * _NALL)


def _score(idx):
    """Return (max multiplicity, target area) for a 30-point index set."""
    if len(idx) != _N:
        return 0, 0
    c = _DIFF_CNT[idx].sum(axis=0)
    # doubled-area of pair (i,j) with base 0 = |cross(P_i, P_j)|
    m = np.maximum.reduce([
        c.ravel(),
        _PAIR_AREA[idx, idx].ravel(),
        _PAIR_AREA[idx, idx].ravel()[::-1],
    ])
    k = int(np.argmax(m))
    return int(m[k]), k


def _fast_move(idx, i, j):
    """Score after removing idx[i] and adding j (j not in idx). O(n)."""
    old = idx[i]
    c = _DIFF_CNT[idx].sum(axis=0)
    c -= _DIFF_CNT[old]
    c += _DIFF_CNT[j]
    new_idx = idx.copy()
    new_idx[i] = j
    m = np.maximum.reduce([
        c.ravel(),
        _PAIR_AREA[new_idx, new_idx].ravel(),
        _PAIR_AREA[new_idx, new_idx].ravel()[::-1],
    ])
    return int(np.max(m))


def _rand_init(rng):
    perm = rng.permutation(_NALL)
    chosen = []
    seen = set()
    for p in perm:
        if p not in seen:
            seen.add(p)
            chosen.append(p)
            if len(chosen) == _N:
                break
    return np.array(chosen, dtype=np.int64)


def _local_search(idx, rng, deadline):
    cur, _ = _score(idx)
    best = cur
    best_idx = idx.copy()
    # precompute per-position candidate lists (fast, vectorized)
    cand = [np.setdiff1d(np.arange(_NALL), idx, assume_unique=True) for _ in range(_N)]
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        improved = False
        # pass 1: try all single-point swaps (cheap)
        for i in range(_N):
            cands = cand[i]
            if cands.size == 0:
                continue
            for j in cands:
                v = _fast_move(idx, i, j)
                if v > cur:
                    old = idx[i]
                    idx[i] = j
                    cand[i] = np.setdiff1d(np.arange(_NALL), idx, assume_unique=True)
                    cand = [c if c == cand[i] else np.setdiff1d(c, np.array([old]), assume_unique=False) if old in c else c for c in cand]
                    # rebuild other candidate lists lazily (cheap enough)
                    for k in range(_N):
                        if k != i:
                            cand[k] = np.setdiff1d(np.arange(_NALL), idx, assume_unique=True)
                    cur = v
                    improved = True
                    if cur > best:
                        best = cur
                        best_idx = idx.copy()
                    break  # restart i-loop after a move
        if not improved:
            # pass 2: a few random 2-point swaps to escape local optima
            for _ in range(200):
                if time.time() >= deadline:
                    break
                i1 = rng.randrange(_N)
                i2 = rng.randrange(_N)
                if i1 == i2:
                    continue
                c1 = cand[i1]
                c2 = cand[i2]
                if c1.size == 0 or c2.size == 0:
                    continue
                j1 = c1[rng.randrange(c1.size)]
                j2 = c2[rng.randrange(c2.size)]
                if j1 == j2 or j1 in idx or j2 in idx:
                    continue
                if j1 == idx[i1] or j2 == idx[i2]:
                    continue
                new_idx = idx.copy()
                new_idx[i1] = j1
                new_idx[i2] = j2
                v, _ = _score(new_idx)
                if v > cur:
                    idx = new_idx
                    cand = [np.setdiff1d(np.arange(_NALL), idx, assume_unique=True) for _ in range(_N)]
                    cur = v
                    improved = True
                    if cur > best:
                        best = cur
                        best_idx = idx.copy()
                    break
            if not improved:
                # random restart of the local search state
                idx = _rand_init(rng)
                cand = [np.setdiff1d(np.arange(_NALL), idx, assume_unique=True) for _ in range(_N)]
                cur, _ = _score(idx)
                if cur > best:
                    best = cur
                    best_idx = idx.copy()
    return best, best_idx


def construct():
    global _T0, _DEADLINE
    start = time.time()
    deadline = start + 84.0  # safety margin
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    # candidate initializations: seed + a few random
    inits = []
    seed_pts = [(x, y) for x in range(6) for y in range(5)]
    seed_idx = np.array([_ALL_IDX[p] for p in seed_pts], dtype=np.int64)
    inits.append(seed_idx)
    for _ in range(4):
        inits.append(_rand_init(np_rng))

    best = 0
    best_idx = seed_idx.copy()
    for idx in inits:
        if time.time() >= deadline:
            break
        v, _ = _score(idx)
        if v > best:
            best = v
            best_idx = idx.copy()
        # local search from this init
        v2, idx2 = _local_search(idx.copy(), np_rng, deadline)
        if v2 > best:
            best = v2
            best_idx = idx2

    # final safety: ensure we have a valid 30 distinct points
    pts = [tuple(_ALL[i]) for i in best_idx]
    if len(set(pts)) != _N:
        pts = [tuple(p) for p in seed_pts]
    return pts
