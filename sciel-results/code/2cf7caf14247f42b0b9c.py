# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 115.0   # explicit safety margin vs 120s kill
    rng = random.Random(12345)

    # Precompute every t-subset as a sorted row matrix (C(v,t) x t)
    T = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    nT = T.shape[0]

    # All k-subsets (candidate blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    nB = len(all_blocks)
    Bmat = np.array(all_blocks, dtype=np.int64) if nB else np.empty((0, k), dtype=np.int64)

    # covmask[b] = boolean vector of t-subsets covered by candidate block b
    covmask = np.zeros((nB, nT), dtype=bool)
    for b in range(nB):
        covmask[b] = (Bmat[b][None, :] == T).any(axis=0)

    def full_greedy(order):
        """Sequential greedy over the given candidate ordering; returns list of block idx."""
        uncovered = np.ones(nT, dtype=bool)
        chosen = []
        for b in order:
            if not uncovered.any():
                break
            m = covmask[b] & uncovered
            c = m.sum()
            if c > 0:
                chosen.append(b)
                uncovered[m] = False
        return chosen

    def local_improve(chosen, max_passes=4):
        """Swap a chosen block for a candidate covering more currently-uncovered t-subsets."""
        chosen = list(chosen)
        if not chosen:
            return chosen
        for _ in range(max_passes):
            if time.time() > deadline - 4.0:
                break
            cov = np.zeros(nT, dtype=np.int8)
            for b in chosen:
                cov[covmask[b]] += 1
            if not (cov == 0).any():
                break
            improved = False
            for i, b in enumerate(chosen):
                if time.time() > deadline - 4.0:
                    break
                m = covmask[b] & (cov == 0)
                gain = m.sum()
                if gain <= 0:
                    continue
                cand = (covmask & (cov == 0)).sum(axis=1)
                cand[b] = -1
                j = int(np.argmax(cand))
                if cand[j] > gain:
                    chosen[i] = j
                    cov[covmask[b]] -= 1
                    cov[covmask[j]] += 1
                    improved = True
                    break
            if not improved:
                break
        return chosen

    best = None
    while time.time() < deadline - 15.0:
        order = list(range(nB))
        rng.shuffle(order)
        chosen = full_greedy(order)
        if not chosen:
            continue
        chosen = local_improve(chosen)
        if best is None or len(chosen) < len(best):
            best = chosen

    if best is None:
        order = list(range(nB))
        rng.shuffle(order)
        best = full_greedy(order)
        best = local_improve(best)

    return [list(all_blocks[b]) for b in best]
# EVOLVE-BLOCK-END
