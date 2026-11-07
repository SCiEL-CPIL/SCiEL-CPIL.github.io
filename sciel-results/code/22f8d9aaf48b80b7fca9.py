"""Packing edge-disjoint Hamiltonian cycles on the fixed 21-vertex host graph."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
BASE_ADJ = [set() for _ in range(N)]
EDGE_SET = set()
for a, b in P['edges']:
    BASE_ADJ[a].add(b)
    BASE_ADJ[b].add(a)
    EDGE_SET.add((a, b) if a < b else (b, a))


def _check_connected(adj, n):
    if n == 0:
        return True
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    count = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == n


def _find_hamiltonian(adj, deadline, rng, max_steps=2000000):
    """Find one Hamiltonian cycle using only edges in adj.
    Returns list of vertices (length N, distinct) or None."""
    n = N
    # quick degree feasibility
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    # connectivity check
    if not _check_connected(adj, n):
        return None

    best_cycle = None
    steps = 0
    start = 0
    path = [start]
    used = [False] * n
    used[start] = True

    def backtrack():
        nonlocal steps, best_cycle
        if best_cycle is not None:
            return True
        if time.time() >= deadline:
            return False
        if len(path) == n:
            cur = path[-1]
            if start in adj[cur]:
                best_cycle = list(path)
                return True
            return False
        cur = path[-1]
        cand = [x for x in adj[cur] if not used[x]]
        if not cand:
            return False

        # Warnsdorff: prefer candidates with fewer forward options
        cand.sort(key=lambda x: (sum(1 for y in adj[x] if not used[y]), rng.random()))

        for x in cand:
            path.append(x)
            used[x] = True
            steps += 1
            if steps > max_steps:
                path.pop()
                used[x] = False
                return False
            if backtrack():
                return True
            path.pop()
            used[x] = False
        return False

    backtrack()
    return best_cycle


def _is_valid_cycles(cycles, all_edges):
    seen = set()
    vset = None
    for cyc in cycles:
        if len(cyc) != N:
            return False
        if len(set(cyc)) != N:
            return False
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            return False
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            if e not in all_edges:
                return False
            if e in seen:
                return False
            seen.add(e)
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under 90s limit
    rng = random.Random(12345)
    remaining = set(EDGE_SET)
    cycles = []

    # seed fallback: one known cycle (full range) if it is valid in the graph
    seed_cycle = list(range(N))
    seed_ok = all((a, b) if a < b else (b, a) in EDGE_SET
                  for a, b in zip(seed_cycle, seed_cycle[1:] + seed_cycle[:1]))
    fallback = [seed_cycle] if seed_ok else []

    # Allocate time: first cycle gets most, later cycles get less
    # Total 63 edges, each Hamiltonian cycle uses 21 edges, max 3 cycles
    time_per_cycle = [35.0, 30.0, 18.0]  # seconds for each cycle attempt

    for cyc_idx in range(3):
        if time.time() >= deadline:
            break
        # Check if remaining edges are sufficient
        if len(remaining) < N:
            break
        # Check degree feasibility
        temp_adj = [set() for _ in range(N)]
        for (u, v) in remaining:
            temp_adj[u].add(v)
            temp_adj[v].add(u)
        feasible = True
        for v in range(N):
            if len(temp_adj[v]) < 2:
                feasible = False
                break
        if not feasible:
            break
        if not _check_connected(temp_adj, N):
            break

        found = None
        # multiple restarts with fresh randomness
        max_restarts = 500
        for attempt in range(max_restarts):
            if time.time() >= deadline:
                break
            found = _find_hamiltonian(temp_adj, deadline, rng)
            if found is not None:
                break
        if found is None:
            break
        # remove its edges from remaining
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
        cycles.append(list(cyc))

    # Validate; if anything is off, fall back.
    if cycles and _is_valid_cycles(cycles, EDGE_SET):
        return [list(c) for c in cycles]
    if fallback:
        return [list(c) for c in fallback]
    # last resort: any valid single cycle via one more attempt
    remaining = set(EDGE_SET)
    for attempt in range(500):
        if time.time() >= deadline:
            break
        f = _find_hamiltonian([BASE_ADJ[v] for v in range(N)], deadline, rng)
        if f is not None and _is_valid_cycles([f], EDGE_SET):
            return [list(f)]
    return [list(range(N))]
# EVOLVE-BLOCK-END
