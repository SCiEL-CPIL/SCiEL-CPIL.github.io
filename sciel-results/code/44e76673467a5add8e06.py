import random
from itertools import combinations
from math import lcm

# Precompute bad triples
bad_triples = []
triples_by_vertex = {i: [] for i in range(1, 129)}

def is_bad_triple(a, b, c):
    if a == b or b == c or a == c:
        return False
    l_ab = lcm(a, b)
    l_ac = lcm(a, c)
    l_bc = lcm(b, c)
    return l_ab == l_ac == l_bc

for a in range(1, 129):
    for b in range(a + 1, 129):
        for c in range(b + 1, 129):
            if is_bad_triple(a, b, c):
                bad_triples.append((a, b, c))
                # Store (u, u_bit, w, w_bit)
                triples_by_vertex[a].append((b, 1 << (b - 1), c, 1 << (c - 1)))
                triples_by_vertex[b].append((a, 1 << (a - 1), c, 1 << (c - 1)))
                triples_by_vertex[c].append((a, 1 << (a - 1), b, 1 << (b - 1)))

def greedy_mis():
    S = set(range(1, 129))
    while True:
        bad_found = None
        for a, b, c in bad_triples:
            if a in S and b in S and c in S:
                bad_found = (a, b, c)
                break
        if not bad_found:
            break
        a, b, c = bad_found
        def degree(v):
            count = 0
            for u, ub, w, wb in triples_by_vertex[v]:
                if u in S and w in S:
                    count += 1
            return count
        da, db, dc = degree(a), degree(b), degree(c)
        if da >= db and da >= dc:
            S.remove(a)
        elif db >= da and db >= dc:
            S.remove(b)
        else:
            S.remove(c)
    return len(S)

initial_best = greedy_mis()

def construct():
    global initial_best
    best_size = initial_best
    best_set_mask = 0
    
    full_mask = (1 << 128) - 1
    
    import sys
    sys.setrecursionlimit(500)

    def find_bad_triple(S_mask):
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            v = idx + 1
            for u, ub, w, wb in triples_by_vertex[v]:
                if (S_mask & ub) and (S_mask & wb):
                    return (v, u, w)
            temp ^= lsb
        return None

    def deg_in_mask(v, mask):
        count = 0
        for u, ub, w, wb in triples_by_vertex[v]:
            if (mask & ub) and (mask & wb):
                count += 1
        return count

    def solve(S_mask, current_size, rem_count):
        nonlocal best_size, best_set_mask
        
        if current_size + rem_count <= best_size:
            return
        
        triple = find_bad_triple(S_mask)
        if triple is None:
            if current_size + rem_count > best_size:
                best_size = current_size + rem_count
                best_set_mask = S_mask
            return
        
        a, b, c = triple
        
        da = deg_in_mask(a, S_mask)
        db = deg_in_mask(b, S_mask)
        dc = deg_in_mask(c, S_mask)
        
        candidates = [(da, a), (db, b), (dc, c)]
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for _, v in candidates:
            v_bit = 1 << (v - 1)
            new_mask = S_mask & ~v_bit
            solve(new_mask, current_size, rem_count - 1)
            
    solve(full_mask, 0, 128)
    
    result = []
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
        
    return result
