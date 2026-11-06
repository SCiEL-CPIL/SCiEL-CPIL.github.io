"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 9
M = 1 << N            # 512 vertices
EDGE = 2 * M          # 1024 directed edges
NEST = 4 * M          # 2048 nesting (u,v,w)
DEADLINE = time.time() + 110.0   # safety margin before the 150 s kill

# ---------------------------------------------------------------------------
# Precompute the induced-path feasibility as bitsets.
#   can_extend[u][v]  = bitmask of w such that {u,v,w} is an induced P3
#                       (i.e. w != u  and  dist(w,u) >= 2).
#   can_extend2[u][v] = can_extend[u][v] & ~bit(v)  (forbid immediate backtrack)
# ---------------------------------------------------------------------------
_can_extend = np.zeros((M, M), dtype=object)
_can_extend2 = np.zeros((M, M), dtype=object)
for u in range(M):
    for v in range(M):
        if (u ^ v).bit_count() != 1:
            continue
        m = 0
        for w in range(M):
            if w == v:
                continue
            if (w ^ u).bit_count() >= 2:
                m |= (1 << w)
        _can_extend[u][v] = m
        _can_extend2[u][v] = m & ~(1 << v)


def construct():
    can1 = _can_extend
    can2 = _can_extend2

    def dfs(prefix, used, limit):
        """Longest-path DFS in the directed line graph.
        prefix: list of directed edges (u,v) already taken (the snake).
        used:   set of directed edges already used.
        Returns the best prefix (list of edges) of length up to `limit`,
        or None if it cannot even take one more edge.
        """
        if time.time() > DEADLINE:
            return None
        u, v = prefix[-1]
        avail = can1[u][v] & ~used
        if avail == 0:
            return None
        if len(prefix) >= limit:
            return prefix
        # fail-first: prefer the candidate with the fewest onward options
        best_w = None
        best_c = 1 << 30
        a = avail
        while a:
            w = (a & -a).bit_length() - 1
            a &= a - 1
            c = (can2[v][w] & ~used).bit_count()
            if c < best_c:
                best_c = c
                best_w = w
                if c <= 1:
                    break
        prefix.append((v, best_w))
        used.add((v, best_w))
        if len(prefix) < limit:
            res = dfs(prefix, used, limit)
            if res is not None:
                return res
        prefix.pop()
        used.remove((v, best_w))
        return None

    # ------------------------------------------------------------------
    # Greedy warm-start: build a long valid snake, then lift to edges.
    # ------------------------------------------------------------------
    def greedy(start):
        path = [start]
        used = {start}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path

    def to_edges(path):
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # ------------------------------------------------------------------
    # Iterative local search: re-run the solver from the best prefix found.
    # ------------------------------------------------------------------
    def improve(prefix):
        best = list(prefix)
        while time.time() < DEADLINE - 1.0:
            used = set(best)
            res = dfs(best, used, len(best) + 1)
            if res is None:
                break
            if len(res) > len(best):
                best = list(res)
            else:
                break
        return best

    # Phase 1: greedy warm-start (a couple of starts)
    warm = []
    for s in (0, (1 << N) - 1):
        p = greedy(s)
        if p:
            warm.append(to_edges(p))
    best = max(warm, key=len) if warm else [(0, 1)]

    # Phase 2: iterative solver-driven improvement
    best = improve(best)

    # Phase 3: try the solver from a few distinct greedy prefixes
    for p in warm:
        if len(p) < len(best):
            continue
        cand = improve(list(p))
        if len(cand) > len(best):
            best = cand

    # ------------------------------------------------------------------
    # Build the vertex path and validate.
    # ------------------------------------------------------------------
    def edges_to_path(edges):
        if not edges:
            return []
        path = [edges[0][0]]
        for u, v in edges:
            path.append(v)
        return path

    path = edges_to_path(best)

    # Validate; fall back to a plain greedy snake if anything is off.
    def valid(path):
        if len(path) < 2:
            return len(path) == 1
        for i in range(len(path) - 1):
            if (path[i] ^ path[i + 1]).bit_count() != 1:
                return False
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True

    if not valid(path):
        path = greedy(0)

    return path
