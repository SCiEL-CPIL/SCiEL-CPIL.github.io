# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = None

def _start_clock():
    global _DEADLINE
    _DEADLINE = time.time() + 140.0  # safety margin under 150s kill

def _expired():
    return time.time() >= _DEADLINE

def _is_q3_free(adj):
    # adj: 16x16 bool matrix. Check every 8-subset for a cube embedding.
    n = 16
    for S in combinations(range(n), 8):
        Sset = set(S)
        # count edges within S
        e = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    e += 1
        if e != 12:
            continue
        # count 4-cycles within S
        four = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    for k in range(len(S)):
                        if k != i and k != j and adj[S[i]][S[k]] and adj[S[j]][S[k]]:
                            for l in range(len(S)):
                                if l != i and l != j and l != k and adj[S[i]][S[l]] and adj[S[j]][S[l]] and adj[S[k]][S[l]]:
                                    four += 1
        if four == 12:
            return False
    return True

def _greedy_add(adj, rng):
    # Try to add edges in random order until blocked by Q3.
    n = 16
    cand = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
    rng.shuffle(cand)
    for (u, v) in cand:
        if _expired():
            break
        adj[u][v] = adj[v][u] = True
        if not _is_q3_free(adj):
            adj[u][v] = adj[v][u] = False
    return adj

def _local_search(adj, rng, iters=200):
    n = 16
    best = sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    for _ in range(iters):
        if _expired():
            break
        # pick a random missing edge to add
        cand = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
        if not cand:
            break
        u, v = rng.choice(cand)
        adj[u][v] = adj[v][u] = True
        if _is_q3_free(adj):
            m = sum(adj[i][j] for i in range(n) for j in range(i+1, n))
            if m > best:
                best = m
        else:
            adj[u][v] = adj[v][u] = False
    return adj

def _edges_from_adj(adj):
    n = 16
    return [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]

def construct():
    _start_clock()
    rng = random.Random(12345)
    n = 16
    # Start from K_{3,13} (39 edges) - known Q3-free
    adj = [[False]*n for _ in range(n)]
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = adj[b][a] = True

    best_adj = [row[:] for row in adj]
    best_m = 39

    # Multi-restart greedy + local search
    while not _expired():
        work = [row[:] for row in best_adj]
        work = _greedy_add(work, rng)
        work = _local_search(work, rng)
        m = sum(work[i][j] for i in range(n) for j in range(i+1, n))
        if m > best_m:
            best_m = m
            best_adj = [row[:] for row in work]
            # continue searching from this improved point
        if _expired():
            break

    return _edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
