import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _score(U):
    U = np.asarray(U, dtype=np.int64)
    if len(U) < 2:
        return 1.0
    # difference set
    diff = U[:, None] - U[None, :]
    n_diff = len(np.unique(diff))
    # sumset
    s = U[:, None] + U[None, :]
    n_sum = len(np.unique(s))
    mx = int(U.max())
    denom = np.log2(2 * mx + 1)
    if denom <= 0:
        return 1.0
    ratio = n_diff / n_sum
    if ratio <= 1.0:
        return 1.0
    return 1.0 + np.log2(ratio) / denom

def _golomb(n):
    # Greedy Golomb ruler
    marks = [0]
    for _ in range(n - 1):
        best = None
        best_len = -1
        for cand in range(marks[-1] + 1, marks[-1] + 200):
            diffs = set()
            ok = True
            for i in range(len(marks)):
                d = cand - marks[i]
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                continue
            # prefer smallest cand that keeps distinct diffs
            best = cand
            break
        if best is None:
            # fallback
            best = marks[-1] + 1
            while True:
                diffs = set()
                ok = True
                for i in range(len(marks)):
                    d = best - marks[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if ok:
                    break
                best += 1
        marks.append(best)
    return np.array(marks, dtype=np.int64)

def construct():
    global _START
    _START = time.time()
    best = np.array([0, 1, 3], dtype=np.int64)
    best_score = _score(best)

    rng = np.random.default_rng(0)

    seeds = [
        np.array([0, 1, 3], dtype=np.int64),
        _golomb(8),
        _golomb(10),
        # base-3 small
        np.array([0, 1, 3, 4, 9, 10, 12, 13], dtype=np.int64),
    ]
    # random seeds
    for _ in range(6):
        n = rng.integers(6, 14)
        mx = rng.integers(20, 60)
        s = np.sort(rng.choice(mx + 1, size=n, replace=False))
        if 0 not in s:
            s[0] = 0
        seeds.append(s.astype(np.int64))

    def local_search(U, max_iter=4000):
        U = np.sort(np.unique(U.astype(np.int64)))
        cur = _score(U)
        if cur > best_score[0]:
            best_score[0] = cur
            best[0] = U.copy()
        cur_arr = U
        for it in range(max_iter):
            if _elapsed() > _DEADLINE - 10:
                break
            # choose a move
            r = rng.random()
            if r < 0.4:
                # add element
                mx = cur_arr.max()
                cand = int(rng.integers(0, mx + 15))
                if cand in cur_arr:
                    continue
                newU = np.sort(np.append(cur_arr, cand))
            elif r < 0.7:
                # remove element (keep 0)
                if len(cur_arr) <= 3:
                    continue
                idx = int(rng.integers(1, len(cur_arr)))
                newU = np.delete(cur_arr, idx)
            else:
                # move element
                if len(cur_arr) <= 3:
                    continue
                idx = int(rng.integers(1, len(cur_arr)))
                old = cur_arr[idx]
                mx = cur_arr.max()
                newval = int(rng.integers(0, mx + 15))
                if newval in cur_arr:
                    continue
                newU = np.sort(np.delete(np.append(cur_arr, newval), idx))
            s = _score(newU)
            if s >= cur:
                cur = s
                cur_arr = newU
                if cur > best_score[0]:
                    best_score[0] = cur
                    best[0] = cur_arr.copy()
        return cur_arr

    best_score = [best_score]
    best = [best]
    for seed in seeds:
        if _elapsed() > _DEADLINE - 15:
            break
        local_search(seed)

    if _elapsed() > _DEADLINE - 5:
        return list(best[0])

    # final polish: more iterations on the best
    local_search(best[0], max_iter=20000)

    return list(best[0])
