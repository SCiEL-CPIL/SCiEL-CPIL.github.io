"""Q3-free graph on 16 vertices: simulated annealing."""
import time
import random
from itertools import combinations

DEADLINE = 130.0

# Precompute
Q3_EDGES = []
for i in range(8):
    for bit in [1, 2, 4]:
        j = i ^ bit
        if j > i:
            Q3_EDGES.append((i, j))

ALL_SUBSETS = list(combinations(range(16), 8))

def count_q3(adj):
    """Count Q3 subgraphs using adjacency matrix adj (16x16, 0/1)."""
    count = 0
    for subset in ALL_SUBSETS:
        # Build sub_adj
        sub_adj = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    sub_adj[i][j] = 1
                    sub_adj[j][i] = 1
        ec = sum(sub_adj[i][j] for i in range(8) for j in range(i+1, 8))
        if ec < 12:
            continue
        # Backtrack
        used = [False]*8
        mp = [-1]*8
        def bt(v):
            if v == 8:
                return True
            for s in range(8):
                if used[s]:
                    continue
                ok = True
                for p in range(v):
                    # Check if v and p are adjacent in Q3
                    if (v, p) in Q3_EDGES or (p, v) in Q3_EDGES:
                        if not sub_adj[mp[p]][s]:
                            ok = False
                            break
                if ok:
                    used[s] = True
                    mp[v] = s
                    if bt(v+1):
                        return True
                    used[s] = False
                    mp[v] = -1
            return False
        if bt(0):
            count += 1
    return count

def construct():
    start = time.time()
    random.seed(42)
    
    all_edges = [(i,j) for i in range(16) for j in range(i+1, 16)]
    random.shuffle(all_edges)
    
    cur = set((a,b) for a in range(3) for b in range(3,16))
    adj = [[0]*16 for _ in range(16)]
    for u,v in cur:
        adj[u][v] = 1
        adj[v][u] = 1
    
    best = set(cur)
    best_n = len(cur)
    
    T = 1.0
    cool = 0.9997
    
    while time.time() - start < DEADLINE:
        if T < 1e-6:
            break
        
        e = random.choice(all_edges)
        u,v = e
        if u > v:
            u,v = v,u
        e = (u,v)
        
        if e in cur:
            cur.remove(e)
            adj[u][v] = 0
            adj[v][u] = 0
            de = -1
        else:
            cur.add(e)
            adj[u][v] = 1
            adj[v][u] = 1
            de = 1
        
        q3 = count_q3(adj)
        
        penalty = 100
        # We don't have prev_q3 easily. Let's just track best Q3-free.
        if q3 == 0 and len(cur) > best_n:
            best = set(cur)
            best_n = len(cur)
        
        # Simple acceptance: if q3==0, keep. If q3>0, revert.
        if q3 > 0:
            if e in cur:
                cur.remove(e)
                adj[u][v] = 0
                adj[v][u] = 0
            else:
                cur.add(e)
                adj[u][v] = 1
                adj[v][u] = 1
        
        T *= cool
    
    return [[u,v] for u,v in best]
