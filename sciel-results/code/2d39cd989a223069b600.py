"""Path decomposition via greedy init + iterative max-matching merge (structural change)."""
import time
from collections import defaultdict
from random import Random

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _greedy_init(edge_set, adj, rng, deadline):
    """Fast greedy: produce maximal simple paths covering all edges."""
    remaining = set(edge_set)
    paths = []
    rem_adj = defaultdict(list)
    for u, v in remaining:
        rem_adj[u].append(v)
        rem_adj[v].append(u)

    def rem_degree(v):
        return len(rem_adj[v]) if v in rem_adj else 0

    while remaining and time.time() < deadline:
        rem_odd = [u for u in rem_adj if rem_degree(u) % 2 == 1]
        if rem_odd:
            candidates = sorted(rem_odd, key=lambda u: -rem_degree(u))[:3]
        else:
            candidates = [u for u in rem_adj if rem_degree(u) > 0]
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda u: -rem_degree(u))[:3]
        start = rng.choice(candidates)

        path = [start]
        in_path = {start}
        current = start
        while True:
            neighbors = [nb for nb in rem_adj[current]
                         if (min(current, nb), max(current, nb)) in remaining]
            if not neighbors:
                break
            non_cycle = [nb for nb in neighbors if nb not in in_path]
            if not non_cycle:
                break
            non_cycle.sort(key=lambda v: -rem_degree(v))
            if rng.random() < 0.25 and len(non_cycle) > 1:
                next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
            else:
                next_v = non_cycle[0]
            key = (min(current, next_v), max(current, next_v))
            remaining.discard(key)
            path.append(next_v)
            in_path.add(next_v)
            if next_v in rem_adj[current]:
                rem_adj[current].remove(next_v)
            if current in rem_adj[next_v]:
                rem_adj[next_v].remove(current)
            if not rem_adj[current]:
                del rem_adj[current]
            if not rem_adj[next_v]:
                del rem_adj[next_v]
            current = next_v
        if len(path) > 1:
            paths.append(path)

    while remaining and time.time() < deadline:
        e = next(iter(remaining))
        u, v = e
        paths.append([u, v])
        remaining.discard(e)
        if v in rem_adj.get(u, []):
            rem_adj[u].remove(v)
        if u in rem_adj.get(v, []):
            rem_adj[v].remove(u)
        if u in rem_adj and not rem_adj[u]:
            del rem_adj[u]
        if v in rem_adj and not rem_adj[v]:
            del rem_adj[v]
    return paths


def _max_matching_bipartite(left, right, adj_map):
    """Kuhn's algorithm. left/right are vertex sets, adj_map[u] = list of neighbors in right.
    Returns list of (u, v) pairs."""
    match_r = {}
    def dfs(u, seen):
        for v in adj_map.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False
    for u in left:
        if dfs(u, set()):
            pass
    return list(match_r.items())  # (v, u) pairs; we want (u, v)


def _try_merge(paths, adj, edge_set):
    """One round: build aux bipartite graph of path endpoints, find max matching,
    apply all merges that are valid (simple + edge exists + not already used)."""
    n_paths = len(paths)
    if n_paths <= 1:
        return paths, False

    # Build set of used edges (frozenset of vertex pairs) for quick lookup
    used_edges = set()
    for p in paths:
        for i in range(len(p) - 1):
            used_edges.add((min(p[i], p[i+1]), max(p[i], p[i+1])))

    # For each path, store its vertex set
    path_vertex_sets = [set(p) for p in paths]

    # Build aux graph: left side = starts (path i, 'L'), right side = ends (path j, 'R')
    # Edge (i_L, j_R) exists if:
    #   i != j, edge (end_i, start_j) in edge_set, not in used_edges,
    #   start_j not in path_vertex_sets[i], end_i not in path_vertex_sets[j]
    # (second check is redundant since end_i is in path i and start_j in path j,
    #  but we need start_j not in path i and end_i not in path j for simplicity after merge)

    left_nodes = list(range(n_paths))   # start of path i
    right_nodes = list(range(n_paths))  # end of path j
    adj_map = defaultdict(list)

    for i in range(n_paths):
        start_i = paths[i][0]
        end_i = paths[i][-1]
        for j in range(n_paths):
            if i == j:
                continue
            start_j = paths[j][0]
            end_j = paths[j][-1]
            # We merge: path_i + path_j reversed? No.
            # Standard merge: take path_i (start_i ... end_i) and path_j (start_j ... end_j)
            # We can connect end_i to start_j if edge exists.
            # Result: start_i ... end_i, start_j ... end_j
            # Need: edge (end_i, start_j) exists and unused.
            # Need: start_j not in path_i's vertices (for simplicity)
            # Need: end_i not in path_j's vertices (redundant, end_i in path_i, start_j in path_j,
            #        but we need the concatenated path to be simple, so no vertex repeats)
            # Actually the concatenation is: path_i vertices then path_j vertices.
            # For simplicity: all vertices in path_i distinct from all in path_j.
            # That's a strong condition. Let's relax: we only need the new edge to connect
            # end_i to start_j, and start_j not already in path_i, and end_i not already in path_j.
            # But if path_i and path_j share any other vertex, the concatenation is not simple.
            # So we need: set(path_i) ∩ set(path_j) = ∅.
            if path_vertex_sets[i] & path_vertex_sets[j]:
                continue
            key = (min(end_i, start_j), max(end_i, start_j))
            if key in edge_set and key not in used_edges:
                adj_map[i].append(j)

    # Find max matching
    # left = starts (i), right = ends (j)
    # adj_map[i] = list of j's
    match_r = {}  # j -> i
    def dfs(u, seen):
        for v in adj_map.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False
    for u in left_nodes:
        dfs(u, set())

    # Apply all matched pairs
    if not match_r:
        return paths, False

    new_paths = []
    merged = set()
    for j, i in match_r.items():
        if i in merged or j in merged:
            continue
        # Merge path_i and path_j: path_i + path_j
        # But wait: we connected end_i to start_j.
        # So new path = path_i + path_j
        # path_i = [start_i, ..., end_i]
        # path_j = [start_j, ..., end_j]
        # New = [start_i, ..., end_i, start_j, ..., end_j]
        # This is valid if the edge (end_i, start_j) exists (checked) and simplicity holds (checked).
        new_path = paths[i] + paths[j]
        new_paths.append(new_path)
        merged.add(i)
        merged.add(j)

    # Add unmerged paths
    for k in range(n_paths):
        if k not in merged:
            new_paths.append(paths[k])

    return new_paths, True


def _hopcroft_karp(left, right, adj_map):
    """Hopcroft-Karp for max bipartite matching. adj_map[u] = list of v in right."""
    # adj_map: u -> [v, ...]
    pair_u = {u: None for u in left}
    pair_v = {v: None for v in right}
    dist = {}

    from collections import deque

    def bfs():
        q = deque()
        for u in left:
            if pair_u[u] is None:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while q:
            u = q.popleft()
            for v in adj_map.get(u, []):
                pu = pair_v.get(v)
                if pu is None:
                    found = True
                elif dist[pu] == float('inf'):
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj_map.get(u, []):
            pu = pair_v.get(v)
            if pu is None or (dist.get(pu, float('inf')) == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching = 0
    while bfs():
        for u in left:
            if pair_u[u] is None:
                if dfs(u):
                    matching += 1
    return matching, pair_u


def decompose_paths(edges, deadline):
    if not edges:
        return []

    edge_set = set()
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)

    degree = {u: len(adj[u]) for u in adj}
    odd_vertices = [u for u in degree if degree[u] % 2 == 1]
    min_paths = max(1, len(odd_vertices) // 2)

    # Multi-start greedy + iterative matching merge
    best_paths = None
    best_count = float('inf')

    max_restarts = 20
    for trial in range(max_restarts):
        if time.time() > deadline:
            break
        rng = Random(trial * 7919 + 13)
        paths = _greedy_init(edge_set, adj, rng, deadline)

        # Iterative matching merge
        for _ in range(50):  # max merge rounds
            if time.time() > deadline:
                break
            new_paths, changed = _try_merge(paths, adj, edge_set)
            if not changed:
                break
            paths = new_paths

        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = paths
        if best_count == min_paths:
            break

    if best_paths is None:
        best_paths = [[u, v] for u, v in edge_set]

    return best_paths


def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit

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
