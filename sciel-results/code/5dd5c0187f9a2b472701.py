import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    supports = list(combinations(range(D), 4))
    N = len(supports)

    # Conflict matrix: True if intersection >= 3
    conflict = np.zeros((N, N), dtype=bool)
    for i in range(N):
        s1 = set(supports[i])
        for j in range(i + 1, N):
            s2 = set(supports[j])
            if len(s1 & s2) >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    def greedy_init(rng):
        avail = np.arange(N)
        selected = []
        in_set = np.zeros(N, dtype=bool)
        while avail.size > 0:
            if time.time() > deadline - 10.0:
                break
            idx = rng.integers(len(avail))
            u = avail[idx]
            selected.append(u)
            in_set[u] = True
            to_remove = np.unique(np.concatenate(([u], np.flatnonzero(conflict[u]))))
            mask = np.isin(avail, to_remove)
            avail = avail[~mask]
        return set(selected)

    def local_search(iset, rng):
        iset = set(iset)
        improved = True
        stall = 0
        while improved and time.time() < deadline - 10.0:
            improved = False
            stall += 1
            sel_list = list(iset)
            rng.shuffle(sel_list)
            
            # 1-2 Swaps
            for v in sel_list:
                if time.time() > deadline - 10.0:
                    break
                iset.remove(v)
                cands = []
                for x in range(N):
                    if x in iset:
                        continue
                    if not np.any(conflict[x][list(iset)]):
                        cands.append(x)
                if len(cands) >= 2:
                    found = False
                    n = len(cands)
                    for i in range(n):
                        u1 = cands[i]
                        for j in range(i + 1, n):
                            u2 = cands[j]
                            if not conflict[u1, u2]:
                                iset.add(u1)
                                iset.add(u2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            if not improved and stall > 10:
                break
        return iset

    best_iset = set()
    best_size = 0
    rng = np.random.default_rng(42)

    for r in range(1000):
        if time.time() > deadline - 15.0:
            break
        iset = greedy_init(rng)
        iset = local_search(iset, rng)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset = set(iset)

    # Final polish
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for x in range(N):
                if x in best_iset:
                    continue
                if not np.any(conflict[x][list(best_iset)]):
                    best_iset.add(x)
                    changed = True

    vectors = []
    for i in range(D):
        for sgn in (1, -1):
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * sgn
            vectors.append(v)
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    return np.array(vectors, dtype=np.int8)
