import time
import random
import bisect

def _valid_add(sums_set, x):
    new_sums = set(sums_set)
    new_sums.add(x)
    for s in sums_set:
        new_sums.add(s + x)
    vals = sorted(new_sums)
    n = len(vals)
    for i in range(n):
        a = vals[i]
        target = 2 * a
        j_start = bisect.bisect_left(vals, target, i + 1)
        for j in range(j_start, n):
            if vals[j] % a == 0:
                return None
    return new_sums

def construct():
    start = time.time()
    deadline = start + 85
    best = []
    orderings = [
        list(range(256, 0, -1)),
        list(range(1, 257)),
        list(range(129, 257)) + list(range(1, 129)),
        list(range(200, 257)) + list(range(1, 200)),
    ]
    for ordering in orderings:
        if time.time() > deadline:
            break
        sums = set()
        cur = []
        for x in ordering:
            if time.time() > deadline:
                break
            ns = _valid_add(sums, x)
            if ns is not None:
                sums = ns
                cur.append(x)
        if len(cur) > len(best):
            best = list(cur)
    random.seed(42)
    for _ in range(300):
        if time.time() > deadline:
            break
        cand = random.sample(range(1, 257), 100)
        random.shuffle(cand)
        sums = set()
        cur = []
        for x in cand:
            if time.time() > deadline:
                break
            ns = _valid_add(sums, x)
            if ns is not None:
                sums = ns
                cur.append(x)
        if len(cur) > len(best):
            best = list(cur)
    if not best:
        best = [31, 32, 34]
    return [int(x) for x in best]
