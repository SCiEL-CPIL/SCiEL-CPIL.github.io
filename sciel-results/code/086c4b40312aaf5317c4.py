"""Decompose three fixed random graphs into few cycles/edges (Erdos-Gallai task).

Strategy: repeatedly extract a 2-factor (a spanning 2-regular subgraph = disjoint
cycles) via max bipartite matching (Hopcroft-Karp) on the current leftover graph,
then extract remaining cycles with randomized longest-walk DFS, and finally emit
any leftover edges as single-edge pieces.  Restart with fresh random edge orderings
within a wall-clock deadline, keeping the best (fewest-piece) valid result.
"""
import time
import random
import sys
from collections import deque

sys.setrecursionlimit(1000000)

SPEC = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
DEADLINE = 112.0  # wall-clock safety margin under the 120 s evaluator limit


def instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


# ---- Hopcroft-Karp on a bipartite graph given by adjacency lists ----
def hk_match(adj, n):
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n

    def bfs():
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return pair_u, matching


def extract_2factor(n, edges, order):
    """Build a 2-factor of the leftover graph via HK; return (cycles, used_edges)."""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    adjL = [list(s) for s in adj]
    pair_u, matching = hk_match(adjL, n)
    if matching < n:
        return [], set()
    # build functional digraph u -> pair_u[u]
    nxt = pair_u[:]
    seen = [0] * n
    cycles = []
    for s in range(n):
        if seen[s]:
            continue
        path = []
        cur = s
        while not seen[cur]:
            seen[cur] = 1
            path.append(cur)
            cur = nxt[cur]
        # cur is already seen; if it is in this path we closed a cycle
        if cur in path:
            idx = path.index(cur)
            cyc = path[idx:]
            if len(cyc) >= 3:
                cycles.append(cyc)
    used = set()
    for cyc in cycles:
        L = len(cyc)
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            used.add((a, b) if a < b else (b, a))
    return cycles, used


def extract_cycles_random(n, edges, rng, time_budget, deadline):
    """Extract cycles from leftover edges via randomized longest-walk DFS."""
    cycles = []
    remaining = set(edges)
    while remaining:
        if time.time() > deadline:
            break
        # start from a random remaining edge, grow a walk
        start_edge = rng.choice(tuple(remaining))
        a, b = start_edge
        # random direction
        if rng.random() < 0.5:
            start, other = a, b
        else:
            start, other = b, a
        # build adjacency of remaining
        adj = {}
        for x, y in remaining:
            adj.setdefault(x, []).append(y)
            adj.setdefault(y, []).append(x)
        # DFS longest path starting at start, first step to other
        best = [start, other]
        stack = [([start, other], {start, other})]
        # iterative DFS with backtracking for a single longest path
        # Use recursion via explicit stack of (path, visited, iterator)
        # Simpler: randomized greedy walk then try to close
        path = [start, other]
        visited = {start, other}
        cur = other
        while True:
            nbrs = [x for x in adj.get(cur, []) if x not in visited]
            if not nbrs:
                break
            cur = rng.choice(nbrs)
            path.append(cur)
            visited.add(cur)
        # try to close: is start adjacent to cur?
        if cur != start and start in adj.get(cur, []):
            if len(path) >= 3:
                cycles.append(path)
                used = set()
                for i in range(len(path)):
                    x = path[i]
                    y = path[(i + 1) % len(path)]
                    used.add((x, y) if x < y else (y, x))
                remaining -= used
        else:
            # cannot close; keep the walk as nothing (edges stay for next round)
            # but to make progress, if the walk used edges, drop them as single edges later
            pass
        if time.time() > deadline:
            break
    return cycles, remaining


def decompose_graph(n, edges, rng, deadline):
    """Return list of pieces (cycles or single edges) covering all edges."""
    # Phase 1: 2-factor
    cycles = []
    remaining = set(edges)
    # try a few random orderings for the 2-factor matching to find a valid one
    for _ in range(8):
        order = list(edges)
        rng.shuffle(order)
        c, used = extract_2factor(n, remaining, order)
        if used:
            cycles.extend(c)
            remaining -= used
            break
    # Phase 2: randomized cycle extraction on remainder
    c2, remaining = extract_cycles_random(n, remaining, rng, 5.0, deadline)
    cycles.extend(c2)
    # Phase 3: leftover single edges
    for e in remaining:
        cycles.append(list(e))
    return cycles


def is_valid(n, edges, pieces):
    edge_set = set((min(a, b), max(a, b)) for a, b in edges)
    seen = set()
    for p in pieces:
        if len(p) == 2:
            u, v = p
            key = (min(u, v), max(u, v))
            if key in seen or key not in edge_set:
                return False
            seen.add(key)
        else:
            if len(p) < 3 or len(set(p)) != len(p):
                return False
            L = len(p)
            for i in range(L):
                u = p[i]
                v = p[(i + 1) % L]
                key = (min(u, v), max(u, v))
                if key in seen or key not in edge_set:
                    return False
                seen.add(key)
    return seen == edge_set


def construct():
    t0 = time.time()
    rng = random.Random(12345)
    results = []
    for (n, p, seed) in SPEC:
        edges = instance(n, p, seed)
        best = None
        # restart with fresh rng seeds until deadline
        attempt = 0
        while time.time() < DEADLINE:
            arng = random.Random(1000 + attempt * 7919)
            pieces = decompose_graph(n, edges, arng, DEADLINE)
            if is_valid(n, edges, pieces):
                if best is None or len(pieces) < len(best):
                    best = pieces
                # early stop if we reached a strong lower bound
                if len(pieces) <= n:  # can't do better than ~n pieces typically
                    break
            attempt += 1
            if attempt >= 40:
                break
        if best is None:
            # safe fallback: every edge as its own piece
            best = [list(e) for e in edges]
        results.append(best)
    return results


if __name__ == "__main__":
    r = construct()
    for i, spec in enumerate(SPEC):
        print(spec, "pieces:", len(r[i]))
    print("total:", sum(len(x) for x in r))
