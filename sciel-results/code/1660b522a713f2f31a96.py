import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
N_WORDS = (N_TRIPLES + 63) // 64  # 9

_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

# Precompute conflict masks
conflicts = [[] for _ in range(N_TRIPLES)]

def _precompute_conflicts():
    for i, (a, b, c) in enumerate(ALL_TRIPLES):
        for a_p in range(V):
            if a_p == a or a_p == b or a_p == c:
                continue
            for b_p in range(V):
                if b_p == a or b_p == b or b_p == c or b_p == a_p:
                    continue
                for c_p in range(V):
                    if c_p == a or c_p == b or c_p == c or c_p == a_p or c_p == b_p:
                        continue
                    t1 = _tidx(a_p, b, c)
                    t2 = _tidx(a, b_p, c)
                    t3 = _tidx(a, b, c_p)
                    t4 = _tidx(a_p, b_p, c)
                    t5 = _tidx(a_p, b, c_p)
                    t6 = _tidx(a, b_p, c_p)
                    t7 = _tidx(a_p, b_p, c_p)
                    
                    mask = [0] * N_WORDS
                    for t in (t1, t2, t3, t4, t5, t6, t7):
                        mask[t >> 6] |= (1 << (t & 63))
                    conflicts[i].append(tuple(mask))

_precompute_conflicts()

def _is_safe(i, presence):
    for mask in conflicts[i]:
        ok = True
        for w in range(N_WORDS):
            if (presence[w] & mask[w]) != mask[w]:
                ok = False
                break
        if ok:
            return False
    return True

def _greedy(order, deadline):
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        if _is_safe(idx, presence):
            presence[idx >> 6] |= (1 << (idx & 63))
            count += 1
    return count, presence

def construct():
    start = time.time()
    deadline = start + 145
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    while time.time() < deadline:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
    return result

random.seed(42)
