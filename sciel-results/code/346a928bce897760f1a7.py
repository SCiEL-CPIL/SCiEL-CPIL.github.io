"""Packing edge-disjoint Hamiltonian cycles on the fixed 21-vertex host graph."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute base adjacency and edge set.
N = P['n']
BASE_ADJ = [set() for _ in range(N)]
EDGE_SET = set()
for a, b in P['edges']:
    BASE_ADJ[a].add(b)
    BASE_ADJ[b].add(a)
    EDGE_SET.add((a, b) if a < b else (b, a))


def _check_connected(adj, n):
    """Check if graph with adjacency adj on n vertices is connected."""
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


def _find_hamiltonian(remaining_edges, deadline, rng, start_vertex=0, max_steps=200000):
    """Find one Hamiltonian cycle using only edges in remaining_edges.
    Returns list of vertices (length N, distinct) or None.
    Warnsdorff-guided backtracking with forced-move detection."""
    n = N
    adj = [set() for _ in range(n)]
    for (u, v) in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
    # quick degree feasibility
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    # connectivity check
    if not _check_connected(adj, n):
        return None

    best_cycle = None
    steps = 0
    start = start_vertex
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

        # Forced-move detection: if any unused vertex has exactly 2 neighbors in adj,
        # both edges must be used in the Hamiltonian cycle.
        # If cur is one neighbor of such a vertex v, and the other neighbor w is unused,
        # then we MUST go to v now (otherwise v can only be reached via w, but we'd
        # need to enter v from w and exit to... wait, actually if deg(v)=2 and both
        # neighbors are unused, we can't visit v from cur unless cur is one of them).
        # Better: if v is unused with deg(v)=2, and one neighbor is cur, the other
        # must also be used to connect to v. So if we don't go to v now, we must
        # eventually come to v from its other neighbor. This means we should prioritize
        # going to v if it's a candidate.
        # Simpler heuristic: Warnsdorff - prefer candidates with fewer forward options
        # Tie-break: prefer candidates that are "forced" (have few options)
        cand_scores = []
        for x in cand:
            fwd = sum(1 for y in adj[x] if not used[y])
            # Add a small bonus if x has exactly 2 total neighbors (forced vertex)
            total_deg = len(adj[x])
            score = fwd
            if total_deg == 2:
                score -= 0.5  # slight preference for forced vertices
            cand_scores.append((score, x))
        
        cand_scores.sort(key=lambda t: t[0])
        # Add randomness to break ties
        for i in range(len(cand_scores) - 1, 0, -1):
            j = rng.randint(0, i)
            cand_scores[i], cand_scores[j] = cand_scores[j], cand_scores[i]
        
        for _, x in cand_scores:
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

    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    # Use different starting vertices for diversity
    start_vertices = [0, 1, 2]

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
        # Try multiple restarts with different starting vertices
        max_restarts = 100
        for attempt in range(max_restarts):
            if time.time() >= deadline:
                break
            # Cycle through starting vertices
            sv = start_vertices[attempt % len(start_vertices)]
            # Add some randomness to the start choice
            if attempt > 0 and rng.random() < 0.3:
                sv = rng.randint(0, N - 1)
            found = _find_hamiltonian(remaining, deadline, rng, start_vertex=sv)
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
    for attempt in range(200):
        if time.time() >= deadline:
            break
        sv = attempt % N
        f = _find_hamiltonian(remaining, deadline, rng, start_vertex=sv)
        if f is not None and _is_valid_cycles([f], EDGE_SET):
            return [list(f)]
    return [list(range(N))]
# EVOLVE-BLOCK-END
