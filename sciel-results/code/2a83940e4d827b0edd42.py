import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0

def _score_of(S):
    S = np.sort(np.asarray(S, dtype=np.int64))
    n = S.size
    if n == 0:
        return -1.0
    maxU = int(S[-1])
    if maxU <= 0:
        return 0.0
    diff = np.unique(S[:, None] - S[None, :])
    summ = np.unique(S[:, None] + S[None, :])
    ratio = diff.size / float(summ.size)
    return 1.0 + np.log(ratio) / np.log(2.0 * maxU + 1)

def _golomb_ruler(n, max_val):
    # Simple greedy construction of a Golomb-like ruler
    a = [0]
    for _ in range(n - 1):
        best = None
        best_pen = None
        for x in range(a[-1] + 1, max_val + 1):
            diffs = [abs(x - y) for y in a]
            pen = sum(1 for d in diffs if d in diffs[:0])  # placeholder
            # count duplicate differences
            dup = len(diffs) - len(set(diffs))
            if best_pen is None or dup < best_pen:
                best_pen = dup
                best = x
                if best_pen == 0:
                    break
        if best is None:
            break
        a.append(best)
    return a

def _greedy_optimize(S, iters, max_val):
    S = list(S)
    cur = _score_of(S)
    n = len(S)
    for it in range(iters):
        if time.time() > _DEADLINE - 5:
            break
        best_move = None
        best_score = cur
        # Try adding a new element
        candidates = []
        for x in range(1, max_val + 1, max(1, max_val // 200)):
            candidates.append(('add', x, 0))
        # Try moving an element
        for i in range(1, n):
            for delta in (-3, -2, -1, 1, 2, 3):
                newv = S[i] + delta
                if newv > 0:
                    candidates.append(('move', i, newv))
        # Try removing an element (not 0)
        for i in range(1, n):
            candidates.append(('del', i, 0))
        for typ, i, val in candidates:
            newS = S[:]
            if typ == 'add':
                if val in newS:
                    continue
                newS.append(val)
            elif typ == 'move':
                if val in newS:
                    continue
                newS[i] = val
            elif typ == 'del':
                newS.pop(i)
            s = _score_of(newS)
            if s > best_score:
                best_score = s
                best_move = (typ, i, val, newS)
        if best_move is None:
            break
        S = best_move[3]
        cur = best_score
    return S, cur

def construct():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    t_start = time.time()
    remaining = _DEADLINE - (t_start - _T0)
    if remaining <= 0:
        return [0, 1, 3]

    # Start from a Golomb ruler seed
    n_seed = 20
    max_val = 500
    S = _golomb_ruler(n_seed, max_val)
    if 0 not in S:
        S.insert(0, 0)
    S = list(set(S))
    S.sort()

    best_S = S[:]
    best_score = _score_of(S)

    # Local search with multiple restarts
    max_val = 1000
    iters_per_run = 200
    while time.time() - t_start < remaining - 10:
        S, sc = _greedy_optimize(best_S[:], iters_per_run, max_val)
        if sc > best_score:
            best_score = sc
            best_S = S
        # Perturb for restart
        if time.time() - t_start < remaining - 20:
            S = best_S[:]
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            for _ in range(3):
                if len(S) > 2:
                    i = rng.integers(1, len(S))
                    S[i] = int(rng.integers(1, max_val))
            S = sorted(set(S))
            if 0 not in S:
                S.insert(0, 0)
            best_S = S

    # Final polish with finer steps
    if time.time() - t_start < remaining - 5:
        S, sc = _greedy_optimize(best_S[:], 300, max_val)
        if sc > best_score:
            best_score = sc
            best_S = S

    return best_S

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
