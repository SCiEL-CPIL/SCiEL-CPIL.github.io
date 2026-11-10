import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N

def _is_sidon(S):
    sums = bytearray(MAX_SUM + 1)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            s = a + S[j]
            if sums[s]:
                return False
            sums[s] = 1
    return True

def _maximize_ascending(S):
    S = sorted(S)
    sums = bytearray(MAX_SUM + 1)
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums[S[i] + S[j]] = 1
    Sset = set(S)
    for x in range(1, N + 1):
        if x in Sset:
            continue
        if sums[2 * x]:
            continue
        ok = True
        for a in S:
            if sums[x + a]:
                ok = False
                break
        if ok:
            S.append(x)
            Sset.add(x)
            sums[2 * x] = 1
            for a in S[:-1]:
                sums[x + a] = 1
    return S

def _cost(S):
    return len(_maximize_ascending(list(S)))

def construct():
    start = time.time()
    deadline = start + 50.0
    random.seed(12345)

    base = _maximize_ascending([])
    best = base
    best_cost = len(base)

    cur = list(base)
    cur_cost = best_cost

    T0 = 2000.0
    T = T0
    alpha = 0.9998
    n_elems = list(range(1, N + 1))

    while time.time() < deadline:
        move = random.random()
        cand = list(cur)
        if move < 0.5:
            if len(cand) > 1:
                cand.pop(random.randrange(len(cand)))
            else:
                continue
        elif move < 0.8:
            if not cand:
                continue
            idx = random.randrange(len(cand))
            removed = cand.pop(idx)
            cands = [x for x in n_elems if x not in cand and x != removed]
            if not cands:
                continue
            cand.append(random.choice(cands))
            if not _is_sidon(cand):
                continue
        else:
            cands = [x for x in n_elems if x not in cand]
            if not cands:
                continue
            cand.append(random.choice(cands))
            if not _is_sidon(cand):
                continue

        cand_cost = _cost(cand)
        if cand_cost < cur_cost:
            cur = cand
            cur_cost = cand_cost
            if cur_cost < best_cost:
                best = _maximize_ascending(list(cur))
                best_cost = cur_cost
        else:
            delta = cand_cost - cur_cost
            if delta > 0 and random.random() < np.exp(-delta / max(T, 1e-9)):
                cur = cand
                cur_cost = cand_cost
        T *= alpha

    return best
