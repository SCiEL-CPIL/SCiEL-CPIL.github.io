"""Rewritten with maximum cycle packing via Hopcroft-Karp on directed line graph."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hopcroft_karp(adj, n_left, n_right):
    """Standard Hopcroft-Karp. adj[u] = list of v in right. 0-indexed."""
    from collections import deque
    INF = 10**9
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
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
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return pair_u, matching

def _decompose_one(edges):
    """Decompose edges into min pieces using max cycle packing + single edges."""
    if not edges:
        return []
    # Build directed arcs: each undirected edge (u,v) with u<v becomes (u,v) and (v,u)
    # arc_id for (u,v): index
    arcs = []
    arc_index = {}
    for (u, v) in edges:
        i = len(arcs)
        arcs.append((u, v))
        arc_index[(u, v)] = i
        j = len(arcs)
        arcs.append((v, u))
        arc_index[(v, u)] = j
    m = len(arcs)  # number of directed arcs = 2 * len(edges)

    # Build adjacency for matching: left node = arc a (as "tail"), right node = arc b (as "head")
    # We want to chain arc a=(x,y) followed by arc b=(y,z). So a->b if a[1] == b[0].
    # Also we cannot chain an arc to its reverse (that would use the same undirected edge twice).
    # arc a and arc b are reverse if a == (x,y) and b == (y,x), i.e., a[0]==b[1] and a[1]==b[0].
    adj = [[] for _ in range(m)]
    for a in range(m):
        x, y = arcs[a]
        for b in range(m):
            if b == a:
                continue
            bx, by = arcs[b]
            if bx != y:
                continue
            # skip reverse (same undirected edge)
            if bx == x and by == y:
                continue
            adj[a].append(b)

    # Run Hopcroft-Karp
    pair_u, _ = _hopcroft_karp(adj, m, m)

    # Build cycles from the permutation: for each arc a, if matched to b, then a->b.
    # A cycle in this functional graph = a sequence of arcs where consecutive arcs share a vertex.
    # Find all directed cycles in the permutation.
    visited = [False] * m
    cycles_arcs = []  # list of lists of arc indices
    for start in range(m):
        if visited[start]:
            continue
        if pair_u[start] == -1:
            visited[start] = True
            continue
        # follow chain
        path = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            path.append(cur)
            nxt = pair_u[cur]
            if nxt == -1:
                break
            cur = nxt
        # Check if path forms a cycle: the last node's match should be the first node
        if path:
            last = path[-1]
            if pair_u[last] == path[0]:
                # It's a cycle
                # But we need to make sure all nodes in path are part of this cycle
                # Actually in a permutation (partial), following from start, if we return to start,
                # the entire path from start to the node that points back to start is the cycle.
                # Let me redo: find the cycle properly.
                pass
        # Better approach: find cycles in the functional graph
    # Let me redo cycle detection properly
    visited = [False] * m
    cycles_arcs = []
    for start in range(m):
        if visited[start]:
            continue
        if pair_u[start] == -1:
            visited[start] = True
            continue
        path = []
        pos_in_path = {}
        cur = start
        while True:
            if visited[cur]:
                break
            if cur in pos_in_path:
                # Found a cycle starting at pos_in_path[cur]
                cycle_start = pos_in_path[cur]
                cycle = path[cycle_start:]
                if len(cycle) >= 3:  # need at least 3 arcs for a valid cycle
                    cycles_arcs.append(cycle)
                for node in path:
                    visited[node] = True
                break
            pos_in_path[cur] = len(path)
            path.append(cur)
            nxt = pair_u[cur]
            if nxt == -1:
                for node in path:
                    visited[node] = True
                break
            cur = nxt

    # Convert arc cycles to vertex cycles
    pieces = []
    used_arcs = set()
    for cycle in cycles_arcs:
        if any(a in used_arcs for a in cycle):
            continue
        # Reconstruct vertex sequence from arcs
        # arc a = (x, y), next arc b = (y, z), etc.
        vertices = []
        for a in cycle:
            x, y = arcs[a]
            if not vertices:
                vertices.append(x)
            else:
                vertices.append(y)
        # Check it's a valid cycle: last vertex should connect back to first
        # The cycle of arcs: a0=(v0,v1), a1=(v1,v2), ..., a_{k-1}=(v_{k-1},v0)
        # So vertices = [v0, v1, v2, ..., v_{k-1}] and closing edge is (v_{k-1}, v0)
        # Verify: for each consecutive pair, the arc exists
        valid = True
        for i in range(len(cycle)):
            a = cycle[i]
            x, y = arcs[a]
            if i == 0:
                if x != vertices[0]:
                    valid = False
                    break
            else:
                if x != vertices[i-1]:
                    valid = False
                    break
            if i == len(cycle) - 1:
                if y != vertices[0]:
                    valid = False
                    break
        if not valid:
            continue
        # Also verify all vertices are distinct (simple cycle)
        if len(set(vertices)) != len(vertices):
            continue
        for a in cycle:
            used_arcs.add(a)
        pieces.append(vertices)

    # Remaining edges: any undirected edge not covered by a cycle becomes a single-edge piece
    covered_undirected = set()
    for a in used_arcs:
        x, y = arcs[a]
        covered_undirected.add(tuple(sorted((x, y))))
    for (u, v) in edges:
        if (u, v) not in covered_undirected:
            pieces.append([u, v])

    return pieces

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback: single edges
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        pieces = _decompose_one(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
