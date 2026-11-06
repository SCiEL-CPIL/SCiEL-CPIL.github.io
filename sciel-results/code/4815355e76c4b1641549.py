# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

N = 256
DEADLINE = 85.0
_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - _START)


def _build_aps():
    """Return list of AP index arrays (0-based). Singletons (step=1,len=1) omitted
    because they never affect the max (|sum|=1)."""
    aps = []
    for step in range(1, N + 1):
        for start in range(0, N - step):
            # maximum length for this start,step
            m = (N - 1 - start) // step + 1
            if m >= 2:
                aps.append(np.arange(start, start + m * step, step))
    return aps


def _build_membership(aps):
    """mem[i] = list of (ap_id, sign_in_ap) for sign position i."""
    mem = [[] for _ in range(N)]
    for aid, arr in enumerate(aps):
        for k, idx in enumerate(arr):
            mem[idx].append((aid, 1 if k % 2 == 0 else -1))
    return mem


def _max_abs(sums):
    return int(np.max(np.abs(sums))) if sums.size else 1


def construct():
    t0 = time.time()
    rng = random.Random(20240611)

    aps = _build_aps()
    M = len(aps)
    mem = _build_membership(aps)

    def sums_from(s):
        out = np.zeros(M, dtype=np.int64)
        for i in range(N):
            si = s[i]
            for aid, sgn in mem[i]:
                out[aid] += si * sgn
        return out

    def flip(s, sums, i):
        delta = -2 * s[i]
        for aid, sgn in mem[i]:
            sums[aid] += delta * sgn
        s[i] = -s[i]

    def flip_set(s, sums, idxs):
        for i in idxs:
            delta = -2 * s[i]
            for aid, sgn in mem[i]:
                sums[aid] += delta * sgn
            s[i] = -s[i]

    def global_rescan(s, sums):
        for i in range(N):
            for aid, sgn in mem[i]:
                sums[aid] = 0
        for i in range(N):
            si = s[i]
            for aid, sgn in mem[i]:
                sums[aid] += si * sgn

    # ---- Phase A: smooth surrogate warm-start (low-step APs dominate the max) ----
    low_step_aps = [a for a in aps if (a[1] - a[0]) <= 16]
    L = len(low_step_aps)
    # for each low AP: (indices, signs)
    lap_idx = [a for a in low_step_aps]
    lap_sign = [np.where(np.arange(len(a)) % 2 == 0, 1.0, -1.0) for a in lap_idx]

    def surrogate_grad(x):
        # soft-max of |sum| over low APs
        g = np.zeros(N, dtype=np.float64)
        for a, sg in zip(lap_idx, lap_sign):
            s = float(np.dot(x[a], sg))
            a_ = abs(s)
            w = math.exp(-a_ / 6.0)
            if s >= 0:
                g[a] += w * sg
            else:
                g[a] -= w * sg
        return g

    # init from a couple random seeds, project
    best_s = None
    best_val = None
    x0 = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(N)], dtype=np.float64)
    # gradient descent on surrogate
    lr = 0.05
    for it in range(4000):
        if _remaining() < 25.0:
            break
        g = surrogate_grad(x0)
        x0 -= lr * g
    s = np.where(x0 >= 0, 1, -1).astype(np.int64)
    sums = sums_from(s)
    val = _max_abs(sums)
    best_s = s.copy()
    best_val = val

    # ---- Phase B: randomized local search with incremental updates ----
    s = best_s.copy()
    sums = sums_from(s)
    cur = best_val

    # order positions by degree (more connected first) for greedy pass
    deg = np.array([len(mem[i]) for i in range(N)], dtype=np.int64)
    order = np.argsort(-deg)

    def greedy_pass(s, sums):
        nonlocal cur
        improved = True
        while improved and _remaining() > 8.0:
            improved = False
            for i in order:
                # try flip
                delta = -2 * s[i]
                # compute change in max without committing: evaluate affected APs
                # cheap exact: temporarily apply, check max, revert if worse
                flip(s, sums, i)
                # quick max over affected only + keep track
                # do a full-ish check via affected set
                affected = [aid for aid, _ in mem[i]]
                mloc = 0
                for aid in affected:
                    mloc = max(mloc, int(abs(sums[aid])))
                # global max may be elsewhere; use stored cur as baseline for unaffected
                if mloc <= cur and mloc <= cur:
                    # could still be fine; accept if local max not exceeding cur
                    # (unaffected max is <= cur by definition of cur)
                    newcur = max(mloc, cur)
                    cur = newcur
                    improved = True
                else:
                    flip(s, sums, i)  # revert
        return cur

    # initial greedy pass
    greedy_pass(s, sums)
    if cur < best_val:
        best_val = cur
        best_s = s.copy()

    # randomized restarts / perturbations to escape plateaus
    base = best_s.copy()
    step = 0
    while _remaining() > 12.0:
        # perturb a random subset of the current best
        s = base.copy()
        k = rng.randint(1, 6)
        idxs = rng.sample(range(N), k)
        sums = sums_from(s)
        flip_set(s, sums, idxs)
        cur = _max_abs(sums)
        if cur < best_val:
            best_val = cur
            base = s.copy()
        # local greedy from here
        greedy_pass(s, sums)
        if cur < best_val:
            best_val = cur
            base = s.copy()
        step += 1
        if step % 50 == 0 and _remaining() > 12.0:
            # occasional deeper multi-flip neighborhood
            s = base.copy()
            sums = sums_from(s)
            for _ in range(3):
                if _remaining() < 12.0:
                    break
                i = rng.randrange(N)
                flip(s, sums, i)
                cur = _max_abs(sums)
                if cur < best_val:
                    best_val = cur
                    base = s.copy()

    # ---- final exact verification of the returned signs ----
    s = base.copy()
    sums = sums_from(s)
    final = _max_abs(sums)
    return [int(v) for v in s]

# EVOLVE-BLOCK-END
