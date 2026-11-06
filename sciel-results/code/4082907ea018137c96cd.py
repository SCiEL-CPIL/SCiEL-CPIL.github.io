"""Dense Q3-free graph on 16 vertices — precomputed greedy."""
import time
import random
from itertools import combinations

N = 16
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]

# Precompute: for each (u,v), the list of 6-subsets from the other 14 vertices
_precomputed = {}
for u, v in ALL_PAIRS:
    others = [x for x in range(N) if x != u and x != v]
    _precomputed[(u, v)] = list(combinations(others, 6))

def _check_q3_containing(adj, u, v):
    """Check if adding edge (u,v) creates a Q3."""
    for other6 in _precomputed[(u, v)]:
        subset = (u, v) + other6
        s = subset
        # Count edges (including new u-v edge)
        ec = 1
        deg = [0] * 8
        # Map vertex to index
        vmap = {s[i]: i for i in range(8)}
        idx_u = vmap[u]
        idx_v = vmap[v]
        deg[idx_u] = 1
        deg[idx_v] = 1
        for i in range(8):
            a = s[i]
            for j in range(i + 1, 8):
                b = s[j]
                if i == idx_u and j == idx_v:
                    continue
                if i == idx_v and j == idx_u:
                    continue
                if (adj[a] >> b) & 1:
                    ec += 1
                    deg[i] += 1
                    deg[j] += 1
        if ec != 12:
            continue
        if any(d != 3 for d in deg):
            continue
        # Bipartite
        color = [-1] * 8
        bip = True
        for si in range(8):
            if color[si] != -1:
                continue
            color[si] = 0
            stack = [si]
            while stack and bip:
                ni = stack.pop()
                nv = s[ni]
                for mj in range(8):
                    if mj == ni:
                        continue
                    if (ni == idx_u and mj == idx_v) or (ni == idx_v and mj == idx_u) or ((adj[nv] >> s[mj]) & 1):
                        if color[mj] == -1:
                            color[mj] = color[ni] ^ 1
                            stack.append(mj)
                        elif color[mj] == color[ni]:
                            bip = False
                            break
        if bip:
            return True
    return False

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    best_edges = []
    best_count = 0
    rng = random.Random(12345)
    
    for trial in range(50):
        if time.time() > DEADLINE:
            break
        
        order = ALL_PAIRS[:]
        rng.shuffle(order)
        
        adj = [0] * N
        cur_edges = []
        
        for u, v in order:
            if _check_q3_containing(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            cur_edges.append([u, v])
        
        if len(cur_edges) > best_count:
            best_count = len(cur_edges)
            best_edges = [list(e) for e in cur_edges]
    
    return best_edges
