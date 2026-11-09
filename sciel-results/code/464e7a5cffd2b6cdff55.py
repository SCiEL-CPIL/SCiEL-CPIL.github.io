import time
import random

N = 256
_START = time.time()
_DEADLINE = _START + 50.0

def _is_sidon(S):
    sums = set()
    arr = sorted(S)
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(i, len(arr)):
            s = ai + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S):
    Sset = set(S)
    for x in range(1, N + 1):
        if x in Sset:
            continue
        if _is_sidon(Sset | {x}):
            return False
    return True

def _greedy(order):
    selected, sums = [], set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _reduce_maximal(S):
    S = list(S)
    changed = True
    while changed and time.time() < _DEADLINE:
        changed = False
        for i in range(len(S)):
            if time.time() >= _DEADLINE:
                break
            cand = S[:i] + S[i + 1:]
            if _is_sidon(cand) and _is_maximal(cand):
                S = cand
                changed = True
                break
    return S

def construct():
    orders = [list(range(1, N + 1))]
    rng = random.Random(0)
    for _ in range(12):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)
    best = None
    for o in orders:
        if time.time() >= _DEADLINE:
            break
        m = _reduce_maximal(_greedy(o))
        if best is None or len(m) < len(best):
            best = m
    return best
