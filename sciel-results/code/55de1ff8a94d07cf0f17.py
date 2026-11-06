import time
import random
import numpy as np
from itertools import combinations, permutations

n = 16

# Precompute
Q3_EDGES = []
for u in range(8):
    for v in range(u+1, 8):
        diff = u ^ v
        if diff & (diff - 1) == 0:
            Q3_EDGES.append((u, v))

Q3_EVEN = [v for v in range(8) if bin(v).count('1') % 2 == 0]
Q3_ODD = [v for v in range(8) if bin(v).count('1') % 2 == 1]

# Precompute index maps
Q3_EVEN_IDX = {v: i for i, v in enumerate(Q3_EVEN)}
Q3_ODD_IDX = {v: i for i, v in enumerate(Q3_ODD)}

def _check_q3_subset(subset, adj):
    """Check if 8-vertex subset induces a graph containing Q3."""
    # Degree filter
    for v in subset:
        cnt = 0
        for u in subset:
            if u != v and (adj[v] >> u) & 1:
                cnt += 1
        if cnt < 3:
            return False
    
    # Try all bipartitions and permutations
    for even_part in combinations(subset, 4):
        even_set = set(even_part)
        odd_part = tuple(v for v in subset if v not in even_set)
        for pe in permutations(even_part):
            for po in permutations(odd_part):
                for cu, cv in Q3_EDGES:
                    if cu in Q3_EVEN_IDX:
                        gu = pe[Q3_EVEN_IDX[cu]]
                    else:
                        gu = po[Q3_ODD_IDX[cu]]
                    if cv in Q3_EVEN_IDX:
                        gv = pe[Q3_EVEN_IDX[cv]]
                    else:
                        gv = po[Q3_ODD_IDX[cv]]
                    if not (adj[gu] >> gv) & 1:
                        break
                else:
                    return True
    return False

def _has_q3(adj):
    """Full Q3 check on 16-vertex graph."""
    for subset in combinations(range(n), 8):
        if _check_q3_subset(subset, adj):
            return True
    return False

def _add_creates_q3(u, v, adj):
    """Check if adding edge (u,v) creates a Q3."""
    remaining = [x for x in range(n) if x != u and x != v]
    for rest in combinations(remaining, 6):
        subset = rest + (u, v)
        if _check_q3_subset(subset, adj):
            return True
    return False

def construct():
    start = time.time()
    deadline = start + 145
    
    # Start with K3,13
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    best_adj = adj[:]
    best_cnt = count_edges(adj)
    cur_adj = adj[:]
    cur_cnt = best_cnt
    
    random.seed(123)
    temp = 3.0
    cooling = 0.999
    
    while temp > 0.01:
        if time.time() > deadline:
            break
        
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        has_edge = bool(adj[u] >> v & 1)
        
        if has_edge:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            delta = -1
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            delta = 1
        
        new_cnt = cur_cnt + delta
        
        if delta > 0 and _add_creates_q3(u, v, adj):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        cur_cnt = new_cnt
        
        if new_cnt > best_cnt:
            best_cnt = new_cnt
            best_adj = adj[:]
        
        if delta < 0:
            prob = np.exp(delta / temp)
            if random.random() > prob:
                # Revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        temp *= cooling
    
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & (-mask)
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            mask &= mask - 1
    
    return edges
