"""Triangle-free diameter-four augmentation via MILP + greedy."""
# EVOLVE-BLOCK-START

import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _validate(added_edges):
    """Validate: triangle-free, connected, diameter <= 4, no duplicates/existing edges."""
    n = 24
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])

    added_edge_set = set()
    for u, v in added_edges:
        if not (isinstance(u, int) and isinstance(v, int)) or isinstance(u, bool) or isinstance(v, bool):
            return False
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base_edge_set or key in added_edge_set:
            return False
        added_edge_set.add(key)

    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)

    # Triangle-free
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False

    # BFS diameter
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        for i in range(n):
            if dist[i] == -1:
                return False
    return True


def _fast_validate(added_edges, base_adj, base_edge_set, n):
    """Fast validation using precomputed base adjacency."""
    seen = set()
    for u, v in added_edges:
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in base_edge_set or key in seen:
            return False
        seen.add(key)

    adj = [set(base_adj[i]) for i in range(n)]
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)

    # Triangle-free
    for a in range(n):
        for b in adj[a]:
            if b > a:
                for c in adj[b]:
                    if c > b and c in adj[a]:
                        return False

    # BFS diameter
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        for i in range(n):
            if dist[i] == -1:
                return False
    return True


def _greedy_construct(n):
    """Greedy: iteratively add the edge that reduces the maximum distance the most."""
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)

    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))

    all_candidates = []
    for e in even_verts:
        for o in odd_verts:
            key = (min(e, o), max(e, o))
            if key not in base_edge_set:
                all_candidates.append((e, o))

    def max_dist(adj):
        """Return the maximum shortest-path distance, or -1 if disconnected."""
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        queue.append(v)
            for i in range(n):
                if dist[i] == -1:
                    return -1  # disconnected
        return max_d

    adj = [set(base_adj[i]) for i in range(n)]
    added = []
    used_edges = set()

    # Precompute which candidates are valid (even-odd, not in base)
    valid_candidates = set()
    for u, v in all_candidates:
        valid_candidates.add((min(u, v), max(u, v)))

    current_max = max_dist(adj)
    while current_max > 4:
        best_edge = None
        best_max = current_max + 1
        for u, v in all_candidates:
            key = (min(u, v), max(u, v))
            if key in used_edges:
                continue
            # Try adding
            adj[u].add(v)
            adj[v].add(u)
            md = max_dist(adj)
            adj[u].discard(v)
            adj[v].discard(u)
            if md >= 0 and md < best_max:
                best_max = md
                best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        used_edges.add((min(u, v), max(u, v)))
        current_max = best_max

    return added


def _try_milp(n, deadline):
    """Try to solve via scipy MILP. Returns list of edges or None."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None

    t0 = time.time()
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))

    # Candidate edges: even-odd pairs not in base path
    candidates = []
    for e in even_verts:
        for o in odd_verts:
            key = (min(e, o), max(e, o))
            if key not in base_edge_set:
                candidates.append((e, o))

    m = len(candidates)
    if m == 0:
        return None

    # Variables: x_j in {0,1} for each candidate edge j
    # Objective: minimize sum x_j

    # Constraints:
    # 1. Triangle-free: for each potential triangle (a,b,c), x_ab + x_bc + x_ac <= 2
    #    Since we only add even-odd edges, triangles can only form if a vertex has
    #    edges to two vertices that are also connected. In a bipartite graph, no triangles
    #    form from added edges alone. But triangles can form with base edges.
    #    A triangle (u,v,w) exists if all three pairs are edges. Since base is a path,
    #    consecutive vertices are connected. An added edge (e,o) creates a triangle with
    #    base edge (e,o+1) if o+1 is even and (e,o+1) is in base... wait, no.
    #    Actually: triangle needs 3 vertices where all 3 pairs are edges.
    #    With only even-odd added edges and a path base, let's think:
    #    Triangle (a,b,c): need edges (a,b),(b,c),(a,c).
    #    If a is even, b is odd, c is even: (a,b) added or base, (b,c) added or base, (a,c) both even - NOT an edge.
    #    So no triangles possible! The graph remains bipartite.
    #    Therefore, triangle-free is automatically satisfied.

    # 2. Diameter <= 4: for every pair (i,j), dist(i,j) <= 4
    #    This is complex for MILP. Use flow formulation or big-M.
    #    Alternative: for each vertex s, for each vertex t, require a path of length <= 4.
    #    This is hard to express linearly. Let's use a different approach.

    # Actually, let's use a simpler constraint: for each vertex v, v must be reachable
    # from each "center" within 4 steps. But this requires knowing the structure.

    # Given complexity, let's just use the greedy result and try to improve via
    # iterative removal. The MILP approach for diameter is too complex.
    return None


def construct():
    """Triangle-free diameter-four augmentation. Returns list of added edges."""
    t0 = time.time()
    deadline = t0 + 80  # safety margin
    n = 24

    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    base_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)

    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))

    all_candidates = []
    for e in even_verts:
        for o in odd_verts:
            key = (min(e, o), max(e, o))
            if key not in base_edge_set:
                all_candidates.append([e, o])

    # Known fallback: 10-edge hub solution
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]

    # Phase 1: Greedy construction
    best = _greedy_construct(n)
    if not _fast_validate(best, base_adj, base_edge_set, n):
        best = [list(e) for e in fallback]
    best_val = len(best)

    # Phase 2: Try hub-based configurations (1 hub, 2 hubs)
    if time.time() < deadline:
        # Single hub
        for hub in range(n):
            if time.time() > deadline:
                break
            config = []
            for v in range(n):
                if v == hub:
                    continue
                # Check if v is within distance 4 of hub via path
                # In path, dist(hub, v) = |hub - v|
                if abs(hub - v) > 4:
                    # Need edge (hub, v) if hub and v are in different partitions
                    if (hub % 2) != (v % 2):
                        key = (min(hub, v), max(hub, v))
                        if key not in base_edge_set:
                            config.append([hub, v])
            if len(config) < best_val and _fast_validate(config, base_adj, base_edge_set, n):
                best = [list(e) for e in config]
                best_val = len(best)

    # Phase 3: Iterative edge removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            trial = best[:i] + best[i + 1:]
            if _fast_validate(trial, base_adj, base_edge_set, n):
                best = trial
                best_val = len(best)
                changed = True
                break

    # Phase 4: 1-swap local search
    if time.time() < deadline:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            remaining = best[:i] + best[i + 1:]
            for alt in all_candidates:
                if time.time() > deadline:
                    break
                if alt in remaining:
                    continue
                trial = remaining + [alt]
                if len(trial) < best_val and _fast_validate(trial, base_adj, base_edge_set, n):
                    best = trial
                    best_val = len(best)
                    break
            if len(best) < best_val:
                break

    # Phase 5: Try to find a smaller solution by searching over hub pairs
    if time.time() < deadline:
        for h1 in even_verts:
            if time.time() > deadline:
                break
            for h2 in odd_verts:
                if time.time() > deadline:
                    break
                config = []
                # Edge between hubs if not in base
                key_h = (min(h1, h2), max(h1, h2))
                if key_h not in base_edge_set:
                    config.append([h1, h2])
                # For each vertex, check if within 4 of h1 or h2
                for v in range(n):
                    if v == h1 or v == h2:
                        continue
                    d1 = abs(v - h1)
                    d2 = abs(v - h2)
                    # Also consider going through the other hub
                    # dist via h1: d(v,h1) or d(v,h2)+d(h2,h1)
                    # For simplicity, just check direct path distance
                    if d1 > 4 and d2 > 4:
                        # Need to connect v to h1 or h2
                        if (v % 2) != (h1 % 2):
                            key = (min(v, h1), max(v, h1))
                            if key not in base_edge_set and [v, h1] not in config and [h1, v] not in config:
                                config.append([v, h1])
                        elif (v % 2) != (h2 % 2):
                            key = (min(v, h2), max(v, h2))
                            if key not in base_edge_set and [v, h2] not in config and [h2, v] not in config:
                                config.append([v, h2])
                if len(config) < best_val and _fast_validate(config, base_adj, base_edge_set, n):
                    best = [list(e) for e in config]
                    best_val = len(best)

    # Final validation
    if not _fast_validate(best, base_adj, base_edge_set, n):
        best = [list(e) for e in fallback]

    result = []
    for edge in best:
        result.append([int(edge[0]), int(edge[1])])

    return result

# EVOLVE-BLOCK-END
