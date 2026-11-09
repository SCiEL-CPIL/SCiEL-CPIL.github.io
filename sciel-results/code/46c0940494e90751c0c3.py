import numpy as np
import time
import random

def _validate(pts):
    pts = np.asarray(pts, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 11:
        return False
    k = pts.shape[0]
    if k < 2:
        return False
    if np.any(pts == 0).all(axis=1):
        return False
    norms = np.einsum('ij,ij->i', pts, pts)
    maxn = int(norms.max())
    # min pair dist^2
    # do it efficiently but exact
    # For large k this is O(k^2 * 11); k up to ~1500 -> ~1.1M*11 ~ ok
    for i in range(k):
        for j in range(i+1, k):
            d = pts[i]-pts[j]
            dsq = int(np.dot(d,d))
            if dsq < maxn:
                return False
    return True

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 540  # safety margin under 630
    # baseline: D_11 roots, R^2=2, k=220
    base = []
    for i in range(d):
        for j in range(i+1, d):
            for si in (1,-1):
                for sj in (1,-1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i]=si; v[j]=sj
                    base.append(v)
    best = np.array(base)
    best_k = best.shape[0]

    # Try R^2=3 greedy packing with randomization, time-bounded
    # candidates: all weight-3 sign vectors
    from itertools import combinations
    cands = []
    for combo in combinations(range(d), 3):
        a,b,c = combo
        for sa in (1,-1):
            for sb in (1,-1):
                for sc in (1,-1):
                    v = np.zeros(d, dtype=np.int64)
                    v[a]=sa; v[b]=sb; v[c]=sc
                    cands.append(v)
    cands = np.array(cands)  # (1320,11)
    # conflict if dot > 1 (i.e., >=2)
    # Greedy: pick order, add if dot<=1 with all accepted
    rng = random.Random(12345)
    cand_idx = np.arange(len(cands))
    best_set = None
    while time.time() < deadline:
        order = rng.sample(range(len(cands)), len(cands))
        accepted = []
        for ci in order:
            v = cands[ci]
            if not accepted:
                accepted.append(ci)
                continue
            # check dot <=1 with all accepted
            ok = True
            for aj in accepted:
                if int(np.dot(v, cands[aj])) > 1:
                    ok = False
                    break
            if ok:
                accepted.append(ci)
        if len(accepted) > best_k:
            best_k = len(accepted)
            best_set = accepted
        # early stop if can't beat
        if best_k >= 1320:
            break
    if best_set is not None:
        cand = np.array([cands[i] for i in best_set])
        if _validate(cand):
            return cand
    # also validate baseline (should be valid)
    return best
