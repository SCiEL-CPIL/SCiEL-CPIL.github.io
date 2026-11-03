import numpy as np
import time
import random
from itertools import combinations

D = 11

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    triples = list(combinations(range(D), 3))
    typeB = []
    for i, j, k in triples:
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0]*D
                    v[i], v[j], v[k] = si, sj, sk
                    typeB.append(v)
    typeB = np.array(typeB, dtype=np.int64)
    nB = len(typeB)

    dots = typeB @ typeB.T
    conflict = dots >= 2

    rng = random.Random(98765)

    best_count = 0
    best_set = None

    def greedy(order):
        available = np.ones(nB, dtype=bool)
        chosen = []
        for idx in order:
            if available[idx]:
                chosen.append(idx)
                available[conflict[idx]] = False
        return chosen

    for restart in range(500000):
        if time.time() > deadline:
            break
        order = list(range(nB))
        rng.shuffle(order)
        chosen = greedy(order)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_set = set(chosen)

    if best_set is not None:
        # fixup: add any now-possible
        chosen = sorted(best_set)
        chosen_set = set(chosen)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(nB):
                if idx in chosen_set:
                    continue
                if not np.any(conflict[idx, chosen]):
                    chosen.append(idx)
                    chosen_set.add(idx)
                    changed = True
                    break

    typeA = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i], v[j] = si, sj
                typeA.append(v)
    typeA = np.array(typeA, dtype=np.int64)

    if best_set:
        chosenB = typeB[sorted(best_set)]
        dotsAB = typeA @ chosenB.T
        a_ok = np.all(dotsAB <= 1, axis=1)
        candA = np.where(a_ok)[0]
        selectedA = []
        for idx in candA:
            v = typeA[idx]
            ok = True
            for c in selectedA:
                if int(v @ typeA[c]) >= 1:
                    ok = False
                    break
            if ok:
                selectedA.append(idx)
        all_vecs = list(chosenB) + [typeA[i] for i in selectedA]
        result = np.array(all_vecs, dtype=np.int64)
    else:
        result = typeB[:1]

    return result
