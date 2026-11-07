"""Rewritten program: search for a minimal triangle-free diameter-4 augmentation of the path."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# Precompute all candidate edges that are triangle-free w.r.t. the path
# In a path, u and v share a neighbor iff they are at distance 2 (|u-v|=2)
# So valid candidates are non-adjacent pairs with |u-v| != 2
_CANDIDATES = []
for u in range(_N):
    for v in range(u + 1, _N):
        if (u, v) in _BASE_SET:
            continue
        if abs(u - v) == 2:
            continue  # shares a neighbor on the path
        _CANDIDATES.append((u, v))

def _build_adj(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_triangle_free(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    return True

def _diameter(added):
    adj = _build_adj(added)
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        if -1 in dist:
            return 10**9
        m = max(dist)
        if m > best:
            best = m
    return best

def _valid(added):
    if not _is_triangle_free(added):
        return False
    d = _diameter(added)
    return d <= P['diameter']

def _shares_neighbor(added, u, v):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return bool(adj[u] & adj[v])

def _construct_greedy(seed):
    rng = random.Random(seed)
    added = []
    added_set = set()
    cur = _diameter(added)
    while cur > P['diameter']:
        cands = []
        for (u, v) in _CANDIDATES:
            if (u, v) in added_set:
                continue
            if _shares_neighbor(added, u, v):
                continue
            cands.append((u, v))
        if not cands:
            return None
        best_edge = None
        best_d = 10**9
        if len(cands) > 300:
            cands = rng.sample(cands, 300)
        for (u, v) in cands:
            trial = added + [(u, v)]
            if not _is_triangle_free(trial):
                continue
            d = _diameter(trial)
            if d < best_d:
                best_d = d
                best_edge = (u, v)
        if best_edge is None:
            return None
        added.append(best_edge)
        added_set.add(best_edge)
        cur = best_d
    return added

def _try_remove(added, seed):
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def _local_search(added, seed, deadline):
    rng = random.Random(seed)
    cur = list(added)
    best = list(cur)
    best_len = len(cur)
    while time.time() < deadline:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                best = list(cur)
                best_len = len(cur)
                break
        if not improved:
            swapped = False
            for i in range(len(cur)):
                trial_remove = cur[:i] + cur[i+1:]
                for (u, v) in _CANDIDATES:
                    if (u, v) in set(trial_remove) or (u, v) in _BASE_SET:
                        continue
                    trial = trial_remove + [(u, v)]
                    if _valid(trial) and len(trial) < len(best):
                        cur = trial
                        best = list(cur)
                        best_len = len(cur)
                        swapped = True
                        break
                if swapped:
                    break
        if not improved and not swapped:
            break
    return best

def construct():
    deadline = time.time() + 55.0
    best = None
    
    for seed in range(50):
        if time.time() >= deadline:
            break
        g = _construct_greedy(seed)
        if g is not None and _valid(g):
            reduced = _try_remove(g, seed)
            if time.time() < deadline - 5:
                reduced = _local_search(reduced, seed, deadline - 2)
            if best is None or len(reduced) < len(best):
                best = reduced
        if best is not None and len(best) <= 5:
            break
    
    if best is None:
        return [[u, v] for u in range(_N) for v in range(u + 1, _N)
                if (u - v) % 2 and (u, v) not in _BASE_SET]
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
