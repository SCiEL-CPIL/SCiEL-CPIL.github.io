import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)

    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    idx_255 = candidates.index(255)
    neighbor_list = [i for i in range(n) if i != idx_255 and (adj[idx_255] >> i) & 1]
    m = len(neighbor_list)

    adj2 = [0] * m
    for i in range(m):
        vi = neighbor_list[i]
        mask = 0
        for j in range(m):
            if i != j and (adj[vi] >> neighbor_list[j]) & 1:
                mask |= (1 << j)
        adj2[i] = mask

    best_size = 0
    best_clique = []

    def expand(P_mask, R, depth=0):
        nonlocal best_size, best_clique
        if time.time() > deadline:
            return
        if P_mask == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return

        # Greedy coloring on P_mask
        P_list = []
        tmp = P_mask
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            P_list.append(v)
            tmp ^= lsb

        # Sort by degree in P
        P_list.sort(key=lambda v: bin(adj2[v] & P_mask).count('1'), reverse=True)

        # Sequential greedy coloring
        color_class = []
        remaining = set(P_list)
        while remaining:
            cls = []
            avail = set(remaining)
            while avail:
                v = min(avail, key=lambda x: bin(adj2[x] & avail).count('1'))
                cls.append(v)
                avail.discard(v)
                to_rm = set()
                for w in avail:
                    if (adj2[v] >> w) & 1:
                        to_rm.add(w)
                avail -= to_rm
            color_class.append(cls)
            for v in cls:
                remaining.discard(v)

        num_colors = len(color_class)
        if len(R) + num_colors <= best_size:
            return

        for cls in reversed(color_class):
            if time.time() > deadline:
                return
            for v in reversed(cls):
                if not (P_mask >> v) & 1:
                    continue
                new_P = P_mask & adj2[v]
                R.append(v)
                expand(new_P, R, depth + 1)
                R.pop()
                P_mask &= ~(1 << v)
                if len(R) + bin(P_mask).count('1') <= best_size:
                    return

    all_P = (1 << m) - 1
    expand(all_P, [])

    result_indices = [idx_255] + [neighbor_list[i] for i in best_clique]
    result = [candidates[i] for i in result_indices]
    result.sort()

    assert 255 in result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert gcd(result[i], result[j]) > 1, f"Failed: gcd({result[i]},{result[j]})=1"

    return result
