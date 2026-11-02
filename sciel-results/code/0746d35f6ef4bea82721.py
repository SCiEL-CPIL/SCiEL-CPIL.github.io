import itertools
import time
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    DEADLINE = 110.0
    pts = range(v)
    # map t-subset -> index
    tsubs = list(itertools.combinations(pts, t))
    t2i = {s: i for i, s in enumerate(tsubs)}
    nT = len(tsubs)

    # all k-subsets
    ksubs = list(itertools.combinations(pts, k))
    nK = len(ksubs)
    # B: (nK, C(k,t)) int32 indices of t-subsets covered by each k-subset
    ckt = len(list(itertools.combinations(range(k), t)))  # C(k,t)
    B = np.empty((nK, ckt), dtype=np.int32)
    for bi, ks in enumerate(ksubs):
        ksset = set(ks)
        for j, ts in enumerate(itertools.combinations(ks, t)):
            B[bi, j] = t2i[ts]

    def greedy(rng, noise):
        cnt = np.zeros(nT, dtype=np.int16)
        chosen = []
        while True:
            if cnt.min() >= 1:
                break
            mask = (cnt == 0)
            scores = mask[B].sum(axis=1).astype(np.float64)
            if noise > 0:
                scores = scores + rng.random(nK) * noise
            best = int(np.argmax(scores))
            # add
            idxs = B[best]
            new = idxs[cnt == 0]
            cnt[new] += 1
            chosen.append(best)
            # early stop on deadline
            if time.time() > DEADLINE:
                break
        return chosen

    def deletion_pass(chosen):
        chosen = list(chosen)
        changed = True
        while changed:
            changed = False
            cnt = np.zeros(nT, dtype=np.int16)
            for b in chosen:
                cnt[B[b]] += 1
            keep = []
            for b in chosen:
                if cnt[B[b]].min() >= 2:
                    cnt[B[b]] -= 1
                    changed = True
                else:
                    keep.append(b)
            chosen = keep
        return chosen

    # baseline deterministic greedy
    rng = np.random.default_rng(0)
    best = None
    best_len = 10**9
    # first deterministic
    ch = greedy(rng, 0.0)
    ch = deletion_pass(ch)
    if len(ch) < best_len:
        best, best_len = ch, len(ch)

    # randomized restarts
    r = 1
    while time.time() < DEADLINE:
        rng = np.random.default_rng(r)
        ch = greedy(rng, 1e-3)
        ch = deletion_pass(ch)
        if len(ch) < best_len:
            best, best_len = ch, len(ch)
        r += 1

    # output
    out = [list(ksubs[b]) for b in best]
    return out
