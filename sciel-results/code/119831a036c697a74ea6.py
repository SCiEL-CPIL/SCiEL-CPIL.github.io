"""Path cover via heuristic construction + exact MILP refinement (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adjacency(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _initial_cover(n, edges, adj):
    # Build a path cover: repeatedly take a longest path through the remaining graph.
    remaining = set(edges)
    remaining_set = set(edges)
    cover = []
    while remaining:
        # pick an edge to start
        start_edge = next(iter(remaining))
        # find a long path: greedily extend from both ends
        path = [start_edge[0], start_edge[1]]
        remaining.discard(start_edge)
        changed = True
        while changed:
            changed = False
            # extend from the tail
            tail = path[-1]
            for nb in adj[tail]:
                if (min(tail, nb), max(tail, nb)) in remaining:
                    path.append(nb)
                    remaining.discard((min(tail, nb), max(tail, nb)))
                    changed = True
                    break
            # extend from the head
            head = path[0]
            for nb in adj[head]:
                if (min(head, nb), max(head, nb)) in remaining:
                    path.insert(0, nb)
                    remaining.discard((min(head, nb), max(head, nb)))
                    changed = True
                    break
        cover.append(path)
    return cover

def _try_milp(n, edges, cover, time_limit):
    """Solve the exact path-cover MILP. Return list of paths or None."""
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        return None

    m = len(edges)
    E = {}
    for i, (u, v) in enumerate(edges):
        E[(u, v)] = i
        E[(v, u)] = i
    V = list(range(n))
    # x_{u,v} for each directed edge, y_v for each vertex
    # objective: minimize sum y
    c = np.zeros(2 * m + n)
    for v in range(n):
        c[2 * m + v] = 1.0

    A = []
    lb = []
    ub = []
    def add_row(coeffs, low, high):
        row = np.zeros(2 * m + n)
        for idx, val in coeffs.items():
            row[idx] = val
        A.append(row)
        lb.append(low)
        ub.append(high)

    # flow conservation: sum_in - sum_out = y_v
    for v in V:
        coeffs = {2 * m + v: 1.0}
        for (u, w) in E:
            if w == v:
                coeffs[(u, w)] = 1.0
            if u == v:
                coeffs[(u, w)] = -1.0
        add_row(coeffs, 0.0, 0.0)

    # x <= y_u and x <= y_v  (i.e. x - y_u <= 0, x - y_v <= 0)
    for (u, w) in E:
        if u == w:
            continue
        coeffs = {(u, w): 1.0, 2 * m + u: -1.0}
        add_row(coeffs, -1e9, 0.0)
        coeffs = {(u, w): 1.0, 2 * m + w: -1.0}
        add_row(coeffs, -1e9, 0.0)

    integrality = np.zeros(2 * m + n)
    integrality[:2 * m] = 1
    integrality[2 * m:] = 1

    constraints = LinearConstraint(np.array(A), lb, ub)
    bounds = Bounds(0.0, 1.0)

    t0 = time.time()
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    except Exception:
        return None
    if not res.success:
        return None

    x = res.x
    # reconstruct paths from selected directed edges
    out_edges = {}
    for (u, w) in E:
        if u == w:
            continue
        if x[(u, w)] > 0.5:
            out_edges[u] = w
    indeg = defaultdict(int)
    for u in out_edges:
        indeg[out_edges[u]] += 1

    paths = []
    for v in V:
        if v in out_edges:
            continue
        # start of a path
        path = [v]
        cur = v
        while cur in out_edges:
            cur = out_edges[cur]
            path.append(cur)
        paths.append(path)
    # verify cover
    used = set()
    for p in paths:
        for a, b in zip(p, p[1:]):
            used.add((min(a, b), max(a, b)))
    if used != set(edges):
        return None
    return paths

def construct():
    deadline = time.time() + 110.0
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        adj = _build_adjacency(edges)
        cover = _initial_cover(n, edges, adj)
        # try to improve with MILP using remaining time
        remaining = deadline - time.time() - 15.0
        if remaining > 1.0:
            improved = _try_milp(n, edges, cover, max(1.0, remaining))
            if improved is not None and len(improved) <= len(cover):
                cover = improved
        results.append(cover)
    return results
# EVOLVE-BLOCK-END
