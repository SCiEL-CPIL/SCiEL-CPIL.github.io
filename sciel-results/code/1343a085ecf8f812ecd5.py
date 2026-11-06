"""Optimized odd covering search over divisors of 51975.

Strategy:
  1. Greedy per-modulus residue init (independent, increasing m).
  2. Smooth surrogate: soft-max weighted coverage, coordinate ascent on residues.
  3. Local-search polish: random coordinate moves with exact delta (coverage
     multiplicity tracked so removal/addition is exact).
Deadline guard returns best found.
"""
import time
import random
import numpy as np

_BASE = 51975
_DIVS = None
_M = None
_IDX = None
_N = None


def _init():
    global _DIVS, _M, _IDX, _N
    if _DIVS is not None:
        return
    divs = [d for d in range(3, _BASE + 1, 2) if _BASE % d == 0]
    divs.sort()
    _DIVS = divs
    _M = np.array(divs, dtype=np.int64)
    _N = len(divs)
    _IDX = [np.arange(0, _BASE, d, dtype=np.int64) for d in divs]


def _final_score(classes):
    cov = np.zeros(_BASE, dtype=bool)
    for m, r in classes:
        cov[int(r) :: int(m)] = True
    return int((~cov).sum())


def construct():
    _init()
    t0 = time.time()
    DEADLINE = t0 + 140.0
    n = _N
    M = _M
    idx = _IDX

    # ---- 1. Greedy init (independent residues) ----
    covered = np.zeros(_BASE, dtype=bool)
    residues = np.zeros(n, dtype=np.int64)
    for i in range(n):
        m = int(M[i])
        arr0 = idx[i]
        best, bestcnt = 0, -1
        for r in range(m):
            cnt = int((~covered[arr0 + r]).sum())
            if cnt > bestcnt:
                bestcnt, best = cnt, r
        residues[i] = best
        covered[arr0 + best] = True

    best_score = int((~covered).sum())
    best_res = residues.copy()

    # ---- 2. Smooth surrogate coordinate ascent ----
    alpha = 1.5
    for sweep in range(4):
        if time.time() > DEADLINE - 30.0:
            break
        cov = np.zeros(_BASE, dtype=bool)
        for i in range(n):
            cov[idx[i] + residues[i]] = True
        cnt = np.array([int((~cov[idx[i] + residues[i]]).sum()) for i in range(n)], dtype=np.float64)
        e = np.exp(alpha * cnt)
        w = e / e.sum()
        order = np.argsort(-w)  # highest-weight first
        for i in order:
            m = int(M[i])
            cur = int(residues[i])
            bestv = cnt[i]
            bestr = cur
            for d in (-1, 1):
                nr = (cur + d) % m
                if nr == cur:
                    continue
                newcnt = int((~cov[idx[i] + nr]).sum())
                if newcnt > bestv:
                    bestv, bestr = newcnt, nr
            if bestr != cur:
                residues[i] = bestr
                cnt[i] = bestv
        cov = np.zeros(_BASE, dtype=bool)
        for i in range(n):
            cov[idx[i] + residues[i]] = True
        s = int((~cov).sum())
        if s < best_score:
            best_score, best_res = s, residues.copy()
        if best_score == 0:
            break

    # ---- 3. Local search polish with exact delta via multiplicity ----
    rng = random.Random(20240517)
    cur = best_res.copy()
    # multiplicity count of coverage per residue
    mult = np.zeros(_BASE, dtype=np.int32)
    for i in range(n):
        mult[idx[i] + cur[i]] += 1
    cur_uncovered = int((mult == 0).sum())

    while time.time() < DEADLINE - 2.0:
        i = rng.randrange(n)
        m = int(M[i])
        cur_r = int(cur[i])
        cands = [(cur_r + d) % m for d in (-1, 1, 2, -2)]
        if rng.random() < 0.25:
            cands.append(rng.randrange(m))
        best_r, best_delta = cur_r, 0
        for nr in cands:
            if nr == cur_r:
                continue
            pos_cur = idx[i] + cur_r
            pos_nr = idx[i] + nr
            # exact delta of uncovered count:
            # removing cur: positions in pos_cur with mult==1 become uncovered (+1 each)
            lost = int((mult[pos_cur] == 1).sum())
            # adding nr: positions in pos_nr with mult==0 become covered (-1 each)
            gained = int((mult[pos_nr] == 0).sum())
            delta = lost - gained
            if delta < best_delta:
                best_delta, best_r = delta, nr
        if best_r != cur_r:
            pos_cur = idx[i] + cur_r
            pos_nr = idx[i] + best_r
            mult[pos_cur] -= 1
            mult[pos_nr] += 1
            cur[i] = best_r
            cur_uncovered += best_delta
            if cur_uncovered < best_score:
                best_score, best_res = cur_uncovered, cur.copy()
                if best_score == 0:
                    break

    best_classes = [[int(M[i]), int(best_res[i])] for i in range(n)]
    # exact final check
    fs = _final_score(best_classes)
    if fs < best_score:
        best_score = fs
    return best_classes
