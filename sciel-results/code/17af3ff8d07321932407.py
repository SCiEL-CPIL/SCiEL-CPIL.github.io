"""Rewritten program: strong edge colouring via DSATUR ordering + bounded backtracking."""
import time
import random
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s wall-clock kill
_t0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _t0)


def _build_instance(n, d, seed):
    rng = random.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        if degree[u] < d and degree[v] < d:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_conflict(edges, adj):
    """conflict[i] = set of edge indices that conflict with edge i.
    Two edges conflict if they share a vertex or are joined by an edge."""
    m = len(edges)
    conflict = [set() for _ in range(m)]
    # For each vertex, all edges incident to it conflict with each other
    incident = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        incident[u].append(i)
        incident[v].append(i)
    for v in incident:
        lst = incident[v]
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                conflict[lst[a]].add(lst[b])
                conflict[lst[b]].add(lst[a])
    # Edges joined by an edge: edge (u,v) and edge (w,x) conflict if (v,w) or (u,x) is an edge
    # i.e., share a vertex (already handled) OR one endpoint of one is adjacent to one endpoint of the other
    # More precisely: edges e1=(a,b), e2=(c,d) are joined if there's an edge (a,c),(a,d),(b,c),(b,d)
    # But sharing a vertex already handled. "Joined by an edge" means there exists an edge connecting
    # a vertex of e1 to a vertex of e2 (and they don't share a vertex).
    for i, (u, v) in enumerate(edges):
        # neighbors of u and v (excluding each other)
        for x in adj.get(u, ()):
            for j in incident.get(x, ()):
                if j != i and (u, v) != edges[j]:
                    conflict[i].add(j)
                    conflict[j].add(i)
        for x in adj.get(v, ()):
            for j in incident.get(x, ()):
                if j != i and (u, v) != edges[j]:
                    conflict[i].add(j)
                    conflict[j].add(i)
    return conflict


def _greedy_color(order, conflict, m):
    """Greedy color in given order. Returns color array or None if fails."""
    color = [-1] * m
    # For each color, track which vertices/edges are "used" — but simpler:
    # a new edge can take color c if no edge of color c conflicts with it.
    # Maintain for each color the set of conflicting edges (union of conflict sets).
    # Faster: for each color, maintain a "blocked" set of edge indices.
    blocked = []  # blocked[c] = set of edge indices that cannot use color c
    for idx in order:
        # find smallest c where idx not in blocked[c]
        c = 0
        while c < len(blocked) and idx in blocked[c]:
            c += 1
        if c == len(blocked):
            blocked.append(set())
        color[idx] = c
        # add all conflicts of idx to blocked[c]
        blocked[c].update(conflict[idx])
    return color


def _solve_instance(n, d, seed):
    edges = _build_instance(n, d, seed)
    m = len(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    conflict = _build_conflict(edges, adj)

    # Degree of each edge in conflict graph
    deg = [len(conflict[i]) for i in range(m)]

    # DSATUR-style greedy: iteratively pick uncolored edge with max saturation,
    # tie-break by conflict degree.
    color = [-1] * m
    used_colors = [set() for _ in range(m)]  # used_colors[c] = set of edges with color c
    remaining = set(range(m))
    # Precompute for speed: for each edge, its neighbors
    order = []
    # We'll do a simple greedy with dynamic ordering
    uncolored = list(range(m))
    # saturation: number of distinct colors used by neighbors
    sat = [0] * m
    color_of_neighbor = [set() for _ in range(m)]

    # Greedy with dynamic ordering
    while uncolored:
        # Pick edge with max saturation, tie-break max degree
        best = max(uncolored, key=lambda i: (len(color_of_neighbor[i]), deg[i]))
        uncolored.remove(best)
        # Find smallest color not used by neighbors
        c = 0
        while c in color_of_neighbor[best]:
            c += 1
        color[best] = c
        for nb in conflict[best]:
            if nb in set(uncolored):
                color_of_neighbor[nb].add(c)
        if _time_left() < 10.0:
            break

    if -1 in color:
        # Fallback: simple greedy in degree order
        color = _greedy_color(sorted(range(m), key=lambda i: -deg[i]), conflict, m)
        if color is None:
            # Last resort: sequential
            color = _greedy_color(list(range(m)), conflict, m)
    return color


def _backtrack_improve(color, conflict, m, budget=5.0):
    """Try to reduce colors via local search / backtracking on a subset."""
    if _time_left() < budget + 1.0:
        return color
    best = list(color)
    best_count = max(best) + 1
    t_start = time.time()

    # Local search: try reassigning each edge to a lower color if valid
    # Build color classes
    for iteration in range(200):
        if time.time() - t_start > budget:
            break
        if _time_left() < 5.0:
            break
        improved = False
        for i in range(m):
            cur = best[i]
            for c in range(cur):
                # Check if i can take color c: no neighbor of i has color c
                ok = True
                for nb in conflict[i]:
                    if best[nb] == c:
                        ok = False
                        break
                if ok:
                    best[i] = c
                    improved = True
                    break
        if not improved:
            break
    # Update best count
    new_count = max(best) + 1
    if new_count < best_count:
        best_count = new_count
    return best


def construct():
    global _t0
    _t0 = time.time()
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _time_left() < 15.0:
            # Not enough time; use fast greedy for remaining
            color = _solve_instance(*spec)
            results.append(color)
            continue
        color = _solve_instance(*spec)
        # Try to improve with local search
        color = _backtrack_improve(color, None, 0, budget=min(8.0, _time_left() - 10.0))
        results.append(color)
    return results
