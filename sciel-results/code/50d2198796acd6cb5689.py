"""Rewritten program: exact branch-and-bound search for minimal triangle-free diameter-4 augmentation."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

# Triangle-free wrt path means: no added edge (u,v) can share a neighbor with any existing edge.
# For a path, u and v share a neighbor iff |u-v| == 2.
# Also, two added edges (a,b) and (c,d) create a triangle iff they share a vertex AND the other endpoints are adjacent in the full graph.
# Since the base is a path (bipartite), and we only add edges between opposite parity, the result is bipartite => triangle-free automatically.
# Wait: path is bipartite (even/odd). Any edge between even and odd keeps it bipartite. Bipartite => triangle-free.
# So the constraint simplifies to: only add edges between vertices of different parity (|u-v| is odd).
# And |u-v| != 1 (not already in path).

_CANDIDATES = []
for u in range(_N):
    for v in range(u + 1, _N):
        if (u, v) in _BASE_SET:
            continue
        if (u - v) % 2 == 0:
            continue  # same parity => would create odd cycle => triangle
        _CANDIDATES.append((u, v))

def _diameter_from_adj(adj):
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

def _diameter(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return _diameter_from_adj(adj)

def _valid(added):
    d = _diameter(added)
    return d <= P['diameter']

def _branch_and_bound(deadline):
    """Exact B&B: try to find solution with k edges for k = 1, 2, 3, ..."""
    # Order candidates by strategic value: prefer edges that connect distant parts of the path
    # Key insight: to get diameter <= 4 from a path of length 23, we need edges that "shortcut" the path.
    # The most useful edges are those connecting vertices far apart on the path.
    # Sort by -|u-v| to try long-range edges first.
    cands = sorted(_CANDIDATES, key=lambda e: -(abs(e[0] - e[1])))
    n_cands = len(cands)
    
    for k in range(1, 15):  # try 1, 2, 3, ... edges
        if time.time() >= deadline:
            break
        # B&B: find if there exists a subset of size k with diameter <= 4
        # Use recursive backtracking with pruning
        found = [None]
        
        def _search(start_idx, chosen, depth):
            if time.time() >= deadline:
                return
            if depth == k:
                if _valid(chosen):
                    found[0] = list(chosen)
                    return
                return
            # Pruning: even if we add all remaining candidates, can we get diameter <= 4?
            # Simple bound: if current partial solution already has diameter > 4 and adding edges can only decrease it,
            # we still need to check. But if we've already exceeded k edges, stop.
            # Better pruning: if the current set of chosen edges already gives diameter <= 4, 
            # then any superset also works (adding edges can't increase diameter).
            # So if _valid(chosen) is already true, we can return immediately with this solution.
            if len(chosen) > 0 and _valid(chosen):
                found[0] = list(chosen)
                return
            
            for i in range(start_idx, n_cands):
                if time.time() >= deadline:
                    return
                if found[0] is not None:
                    return
                chosen.append(cands[i])
                _search(i + 1, chosen, depth + 1)
                chosen.pop()
                if found[0] is not None:
                    return
        
        _search(0, [], 0)
        if found[0] is not None:
            return found[0]
    return None

def _greedy_fallback(deadline):
    """Fast greedy to get an initial solution."""
    added = []
    cur_d = _diameter(added)
    while cur_d > P['diameter'] and time.time() < deadline:
        best_edge = None
        best_d = 10**9
        for (u, v) in _CANDIDATES:
            if (u, v) in set(added):
                continue
            trial = added + [(u, v)]
            d = _diameter(trial)
            if d < best_d:
                best_d = d
                best_edge = (u, v)
        if best_edge is None:
            break
        added.append(best_edge)
        cur_d = best_d
    return added

def construct():
    deadline = time.time() + 55.0
    
    # First, get a quick greedy solution to know an upper bound
    greedy_sol = _greedy_fallback(deadline - 10)
    upper_bound = len(greedy_sol) if greedy_sol else 14
    
    # Now do exact B&B starting from k=1
    exact_sol = _branch_and_bound(deadline)
    
    if exact_sol is not None:
        result = exact_sol
    elif greedy_sol is not None:
        result = greedy_sol
    else:
        # Fallback: add all bipartite non-adjacent edges
        result = [(u, v) for (u, v) in _CANDIDATES]
    
    return [[int(u), int(v)] for (u, v) in result]
# EVOLVE-BLOCK-END
