"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

import random
import time
import sys

_N = P['n']
_EDGE_ORDER = P['edge_order']
_N_EDGES = len(_EDGE_ORDER)

# Precompute edge -> position (1-based)
_EDGE_POS = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    if _a > _b:
        _a, _b = _b, _a
    _EDGE_POS[(_a, _b)] = _i

def _epos(a, b):
    if a > b:
        a, b = b, a
    return _EDGE_POS[(a, b)]

def _cycle_score(cycle):
    # cycle is a list of n vertices; edges are (cycle[i], cycle[(i+1)%n])
    mx = 0
    for i in range(_N):
        u = cycle[i]
        v = cycle[(i + 1) % _N]
        p = _epos(u, v)
        if p > mx:
            mx = p
    return mx

def _find_cycle_in_prefix(K, rng, time_limit, max_nodes=50_000_000):
    """Find a Hamilton cycle using only edges with position <= K.
    Returns cycle list or None."""
    # Build adjacency: for each vertex, list of neighbors with edge pos <= K
    adj = [[] for _ in range(_N)]
    for _a, _b in _EDGE_ORDER:
        if _EDGE_POS[(_a, _b) if _a < _b else (_b, _a)] > K:
            break
        adj[_a].append(_b)
        adj[_b].append(_a)

    # Check minimum degree
    for i in range(_N):
        if len(adj[i]) < 2:
            return None

    # Build sorted adjacency by position (prefer earlier edges)
    for i in range(_N):
        adj[i].sort(key=lambda v: _epos(i, v))

    # Backtracking with bitmasks
    full_mask = (1 << _N) - 1
    start = 0
    best_cycle = [None]

    def backtrack(path, used_mask, current):
        if len(path) == _N:
            # Check closing edge
            if _epos(current, start) <= K:
                best_cycle[0] = list(path)
                return True
            return False

        # Remaining vertices
        remaining = full_mask ^ used_mask
        # Prune: check that remaining vertices can form a path
        # Simple degree check: for each unvisited vertex, count edges to unvisited or to current
        for v in range(_N):
            if remaining & (1 << v):
                # v must connect to at least one unvisited vertex or to start (if v is last)
                has_conn = False
                for w in adj[v]:
                    if w == current:
                        has_conn = True
                        break
                    if remaining & (1 << w):
                        has_conn = True
                        break
                if not has_conn:
                    return False

        # Try neighbors of current
        for nxt in adj[current]:
            if used_mask & (1 << nxt):
                continue
            # If nxt is start and not at the end, skip
            if nxt == start and len(path) < _N - 1:
                continue
            path.append(nxt)
            if backtrack(path, used_mask | (1 << nxt), nxt):
                return True
            path.pop()
        return False

    start_time = time.time()
    # Try with start=0
    path = [start]
    if backtrack(path, 1 << start, start):
        return best_cycle[0]

    # If failed, try random restarts with different vertex orderings
    deadline = time_limit
    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        # Shuffle adjacency order
        for i in range(_N):
            rng.shuffle(adj[i])
        path = [start]
        if backtrack(path, 1 << start, start):
            return best_cycle[0]

    return None

def _two_opt_improve(cycle, rng, time_limit):
    """Try to reduce the max edge position via 2-opt moves."""
    cycle = list(cycle)
    current_score = _cycle_score(cycle)
    best_cycle = list(cycle)
    best_score = current_score

    deadline = time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(_N):
            if time.time() >= deadline:
                break
            for j in range(i + 2, _N):
                if i == 0 and j == _N - 1:
                    continue  # same cycle
                # Reverse segment [i+1..j]
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                ns = _cycle_score(new_cycle)
                if ns < best_score:
                    best_score = ns
                    best_cycle = new_cycle
                    cycle = new_cycle
                    improved = True

    return best_cycle, best_score

def construct():
    rng = random.Random(614)
    start_time = time.time()
    total_deadline = start_time + 75.0  # safety margin for 90s kill

    # Phase 1: Binary search for minimal K with a Hamilton cycle
    # Lower bound: need at least 24 edges, each vertex degree >= 2
    # Upper bound: full graph
    lo = 24
    hi = _N_EDGES

    # First, find any feasible K with a generous time budget
    search_deadline = start_time + 55.0

    # Quick check: can we find a cycle in the full graph?
    cycle = _find_cycle_in_prefix(hi, rng, start_time + 10.0)
    if cycle is None:
        # Shouldn't happen for complete graph, but fallback
        return list(range(_N))

    # Binary search
    best_cycle = cycle
    best_K = hi

    while lo < hi and time.time() < search_deadline:
        mid = (lo + hi) // 2
        c = _find_cycle_in_prefix(mid, rng, time.time() + 20.0)
        if c is not None:
            best_cycle = c
            best_K = mid
            hi = mid
        else:
            lo = mid + 1

    # Phase 2: Local improvement on the best cycle found
    if time.time() < total_deadline - 5.0:
        improved_cycle, improved_score = _two_opt_improve(
            best_cycle, rng, total_deadline - 5.0
        )
        if improved_score < _cycle_score(best_cycle):
            best_cycle = improved_cycle

    return best_cycle
# EVOLVE-BLOCK-END
