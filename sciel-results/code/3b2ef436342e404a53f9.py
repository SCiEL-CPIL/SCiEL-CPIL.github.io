"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

N = P['n']
ADJ = [set() for _ in range(N)]
for a, b in P['edges']:
    ADJ[a].add(b)
    ADJ[b].add(a)
MINV = P['minimum_vertices']

def _score(sel):
    deg = []
    for v in sel:
        c = 0
        for u in ADJ[v]:
            if u in sel:
                c += 1
        deg.append(c)
    return len(set(deg))

def _fast_score(sel, selset):
    deg = []
    for v in sel:
        c = 0
        for u in ADJ[v]:
            if u in selset:
                c += 1
        deg.append(c)
    return len(set(deg))

def construct():
    deadline = time.time() + 80.0
    best_sel = list(range(MINV))
    best_val = _score(best_sel)

    # 1) Local search over the 8-vertex seed
    def local_search(start):
        sel = set(start)
        cur = _score(list(sel), sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in list(sel):
                for b in range(N):
                    if b in sel:
                        continue
                    new = set(sel)
                    new.discard(a)
                    new.add(b)
                    val = _fast_score(list(new), new)
                    if val > cur:
                        sel = new
                        cur = val
                        improved = True
                        break
                if improved:
                    break
        return list(sel), cur

    s, v = local_search(best_sel)
    if v > best_val:
        best_val, best_sel = v, s

    # 2) Exhaustive search over all 8-vertex subsets (22314)
    for combo in combinations(range(N), MINV):
        if time.time() > deadline:
            break
        selset = set(combo)
        val = _fast_score(list(combo), selset)
        if val > best_val:
            best_val = val
            best_sel = list(combo)
            if best_val >= 10:
                break

    # 3) If we found 10, try to extend to 9/10 vertices to maybe push higher
    if best_val >= 10:
        for size in (9, 10):
            for combo in combinations(range(N), size):
                if time.time() > deadline:
                    break
                selset = set(combo)
                val = _fast_score(list(combo), selset)
                if val > best_val:
                    best_val = val
                    best_sel = list(combo)

    # 4) Final local polish
    s, v = local_search(best_sel)
    if v > best_val:
        best_val, best_sel = v, s

    out = []
    for x in best_sel:
        out.append(int(x))
    return out
# EVOLVE-BLOCK-END
