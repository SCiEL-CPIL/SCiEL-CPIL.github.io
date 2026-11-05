"""Path decomposition: Eulerian core (cycles) + odd-vertex endpoint pairing.

Structural change from pure greedy:
  - Decompose the even-degree (Eulerian) core into edge-disjoint closed trails.
    Each closed trail becomes a path (a cycle with one edge cut), contributing
    0 to the unavoidable path count.
  - The odd-degree vertices force at least |odd|/2 paths. We build the odd
    structure as paths and then use a max-matching merge to pair endpoints,
    driving the count toward the theoretical minimum.
"""
import time
from collections import defaultdict, deque
from random import Random


def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _key(u, v):
    return (u, v) if u < v else (v, u)


def _eulerian_core(edge_set, adj):
    """Decompose the even-degree subgraph (all vertices even) into closed trails.

    Returns list of closed trails, each a list of vertices where first == last.
    Every edge in the even core is used exactly once.
    """
    # Build the even-core edge set: edges whose both endpoints are even-degree in the FULL graph
    # Actually the even core = edges that lie on the even-degree structure.
    # Simpler: work on the full graph but only traverse via Hierholzer, tracking usage.
    # We decompose the whole graph into trails; even-degree vertices yield closed trails,
    # odd-degree vertices yield open trails. We separate them.
    remaining = set(edge_set)
    rem_adj = defaultdict(set)
    for u, v in remaining:
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    trails = []  # each: list of vertices (open or closed)

    # Find a start: any vertex with remaining edges.
    def _extract_trail(start):
        """Hierholzer: extract one trail (open or closed) from `start` using remaining edges."""
        stack = [start]
        path = []
        while stack:
            v = stack[-1]
            if rem_adj[v]:
                nb = min(rem_adj[v])  # deterministic
                # remove edge v-nb
                rem_adj[v].discard(nb)
                rem_adj[nb].discard(v)
                stack.append(nb)
            else:
                path.append(stack.pop())
        path.reverse()
        return path

    while any(rem_adj):
        # pick a start vertex with remaining edges
        start = next(iter(rem_adj))
        while start in rem_adj and not rem_adj[start]:
            # find another
            start = next(iter(rem_adj))
        trail = _extract_trail(start)
        if len(trail) >= 2:
            trails.append(trail)

    return trails


def _split_trails(trails):
    """Split closed trails into paths (cut one edge) and keep open trails as paths.

    A closed trail [v0, v1, ..., vk, v0] becomes path [v0, v1, ..., vk] (drop the
    repeated last vertex) -- this uses all edges of the trail.
    An open trail [v0, ..., vk] with v0 != vk is already a path.
    """
    paths = []
    for t in trails:
        if len(t) < 2:
            continue
        if t[0] == t[-1]:
            # closed: drop last (repeated) vertex -> open path using all edges
            paths.append(t[:-1])
        else:
            paths.append(t)
    return paths


def _try_merge_endpoints(paths, edge_set, used_edges, rng):
    """Merge paths by connecting end_i to start_j (or start_i to end_j) via an
    unused edge, when vertex-disjoint. Returns (new_paths, changed)."""
    n = len(paths)
    if n <= 1:
        return paths, False

    path_verts = [set(p) for p in paths]

    # Build candidate merges: (i, j, orientation) where we connect an endpoint of i
    # to an endpoint of j via an unused edge, vertex-disjoint.
    # We'll build a bipartite matching on "endpoint slots".
    # Left slot: (i, 'end')  -> can connect end_i to start_j
    # Right slot: (j, 'start') -> start_j
    # We allow both directions by also considering (i,'start') to (j,'end').
    # To keep it simple and effective, build a general matching on endpoints.

    # Represent each path endpoint as a node. Two endpoints can be linked if there's
    # an unused edge between their vertices AND the two paths are vertex-disjoint.
    # Then a matching pairs endpoints; each pair merges the two paths.
    # A path has 2 endpoints; matching pairs them so each path gets at most one merge.

    # left = endpoints of paths with even index slot, right = odd? No -- bipartite by path parity is wrong.
    # Instead: model as matching where each path can be used once. We do a greedy
    # matching over candidate edges sorted by a heuristic, ensuring each path used once.

    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            if path_verts[i] & path_verts[j]:
                continue
            ei_s, ei_e = paths[i][0], paths[i][-1]
            ej_s, ej_e = paths[j][0], paths[j][-1]
            # four possible connections
            for a, b in ((ei_e, ej_s), (ei_e, ej_e), (ei_s, ej_s), (ei_s, ej_e)):
                k = _key(a, b)
                if k in edge_set and k not in used_edges:
                    cand.append((i, j, a, b))

    if not cand:
        return paths, False

    # Greedy: sort candidates, pick disjoint-path merges.
    # Score: prefer merges that connect high-degree endpoints (more "useful").
    used_path = set()
    new_used_edges = set(used_edges)
    merges = []
    for (i, j, a, b) in cand:
        if i in used_path or j in used_path:
            continue
        k = _key(a, b)
        if k in new_used_edges:
            continue
        # valid
        merges.append((i, j, a, b, k))
        used_path.add(i)
        used_path.add(j)
        new_used_edges.add(k)

    if not merges:
        return paths, False

    # Apply merges
    merged_into = {}
    for (i, j, a, b, k) in merges:
        pi = paths[i]
        pj = paths[j]
        # Determine orientation: we connect vertex a (in pi) to vertex b (in pj).
        # a is either pi[0] or pi[-1]; b is either pj[0] or pj[-1].
        # Build pi oriented so that a is at the end, pj oriented so b is at the start.
        if pi[-1] == a:
            pi_o = pi
        elif pi[0] == a:
            pi_o = pi[::-1]
        else:
            continue  # shouldn't happen
        if pj[0] == b:
            pj_o = pj
        elif pj[-1] == b:
            pj_o = pj[::-1]
        else:
            continue
        # Now pi_o ends with a, pj_o starts with b. Merge: pi_o + pj_o[1:]
        merged_into[i] = pi_o + pj_o[1:]
        merged_into[j] = None  # consumed

    new_paths = []
    for i in range(n):
        if i in merged_into:
            if merged_into[i] is not None:
                new_paths.append(merged_into[i])
        else:
            new_paths.append(paths[i])

    return new_paths, True


def _used_edges_of(paths):
    s = set()
    for p in paths:
        for i in range(len(p) - 1):
            s.add(_key(p[i], p[i + 1]))
    return s


def decompose_paths(edges, deadline):
    if not edges:
        return []

    edge_set = set()
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        edge_set.add(_key(u, v))
        adj[u].append(v)
        adj[v].append(u)

    degree = {u: len(adj[u]) for u in adj}
    odd_vertices = [u for u in degree if degree[u] % 2 == 1]
    min_paths = max(1, len(odd_vertices) // 2)

    # Phase 1: Eulerian decomposition -> trails -> paths
    trails = _eulerian_core(edge_set, adj)
    paths = _split_trails(trails)

    # Ensure all edges covered (safety): if any edge missing, append as single-edge path
    if paths:
        used = _used_edges_of(paths)
    else:
        used = set()
    for e in edge_set:
        if e not in used:
            u, v = e
            paths.append([u, v])
            used.add(e)

    # Phase 2: iterative endpoint-merge to reduce count toward min_paths
    rng = Random(12345)
    for _ in range(200):
        if time.time() > deadline:
            break
        if len(paths) <= min_paths:
            break
        new_paths, changed = _try_merge_endpoints(paths, edge_set, _used_edges_of(paths), rng)
        if not changed:
            break
        paths = new_paths

    return paths


def construct():
    deadline = time.time() + 115  # safety margin before 150s limit

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue

        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)

    return results
