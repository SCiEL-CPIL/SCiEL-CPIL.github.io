"""Rewritten design: direct-constraint local search for dense valid graph."""
import time
import random

N = 15
DEADLINE = 80.0

def _degs(adj):
    return [sum(adj[i]) for i in range(N)]

def _is_valid(adj):
    degs = _degs(adj)
    for u in range(N):
        for v in range(u+1, N):
            if not adj[u][v]:
                continue
            # middle edge (u,v)
            su = [x for x in range(N) if x != v and adj[u][x]]
            tv = [x for x in range(N) if x != u and adj[v][x]]
            deg_su = {}
            for x in su:
                deg_su.setdefault(degs[x], []).append(x)
            for x in tv:
                d = degs[x]
                if d in deg_su:
                    # need s != t
                    for s in deg_su[d]:
                        if s != x:
                            return False
    return True

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _try_add(adj, i, j):
    if i == j or adj[i][j]:
        return False
    adj[i][j] = adj[j][i] = True
    ok = _is_valid(adj)
    if not ok:
        adj[i][j] = adj[j][i] = False
    return ok

def _construct_initial(seed):
    rng = random.Random(seed)
    adj = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < 0.45:
                adj[i][j] = adj[j][i] = True
    # prune: remove edges causing violations
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    adj[i][j] = adj[j][i] = False
                    if not _is_valid(adj):
                        adj[i][j] = adj[j][i] = True
                    else:
                        changed = True
    return adj

def _local_search(adj, rng, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [(i,j) for i in range(N) for j in range(i+1,N) if not adj[i][j]]
        rng.shuffle(missing)
        for i,j in missing:
            if time.time() > deadline:
                return adj
            if _try_add(adj, i, j):
                improved = True
    # swaps
    for _ in range(300):
        if time.time() > deadline:
            break
        cur = _edge_count(adj)
        present = [(i,j) for i in range(N) for j in range(i+1,N) if adj[i][j]]
        missing = [(i,j) for i in range(N) for j in range(i+1,N) if not adj[i][j]]
        if not present or not missing:
            break
        rng.shuffle(present)
        rng.shuffle(missing)
        found = False
        for (a,b) in present[:40]:
            for (c,d) in missing[:40]:
                if time.time() > deadline:
                    return adj
                adj[a][b] = adj[b][a] = False
                if c == d or adj[c][d]:
                    adj[a][b] = adj[b][a] = True
                    continue
                adj[c][d] = adj[d][c] = True
                if _is_valid(adj) and _edge_count(adj) > cur:
                    found = True
                    break
                adj[c][d] = adj[d][c] = False
                adj[a][b] = adj[b][a] = True
            if found:
                break
        if not found:
            break
    return adj

def construct():
    start = time.time()
    best_adj = None
    best_count = -1
    for seed in range(300):
        if time.time() > DEADLINE - 3:
            break
        adj = _construct_initial(seed)
        adj = _local_search(adj, random.Random(seed*7+1), DEADLINE - 3)
        c = _edge_count(adj)
        if c > best_count:
            best_count = c
            best_adj = [row[:] for row in adj]
    if best_adj is None:
        best_adj = [[False]*N for _ in range(N)]
    if not _is_valid(best_adj):
        return []
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    return edges
