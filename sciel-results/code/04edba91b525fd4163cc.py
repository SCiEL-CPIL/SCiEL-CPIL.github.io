import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def min_path_cover_matching(path_edges):
        """
        Given a set of edges, find minimum number of paths covering all edges.
        Uses the vertex-splitting transformation + max matching.
        """
        if not path_edges:
            return []
        # Build adjacency for the subgraph
        adj = defaultdict(set)
        for u, v in path_edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build bipartite graph: for each edge (u,v), create node (u,v)
        # Left nodes: (u,v) can connect to right nodes (v,w) where w != u
        # This is the standard edge-to-edge transition
        left_nodes = []
        left_idx = {}
        for e in path_edges:
            u, v = e
            left_nodes.append(e)
            left_idx[e] = len(left_nodes) - 1

        # For each left node (u,v), it can transition to right node (v,w)
        # where (v,w) is in path_edges and w != u
        right_idx = {}
        for e in path_edges:
            u, v = e
            right_idx[e] = left_idx[e]  # same index space

        # Build bipartite adjacency: left (u,v) -> right (v,w) if w != u
        bip_adj = defaultdict(list)
        for e in path_edges:
            u, v = e
            li = left_idx[e]
            for w in adj[v]:
                if w == u:
                    continue
                ne = tuple(sorted((v, w)))
                if ne in path_edges:
                    ri = left_idx[ne]
                    bip_adj[li].append(ri)

        # Hopcroft-Karp for max bipartite matching
        n_left = len(left_nodes)
        n_right = n_left
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        from collections import deque

        def bfs():
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while q:
                u = q.popleft()
                for v in bip_adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == float('inf'):
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in bip_adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False

        matching_size = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1

        # Number of paths = |E| - |matching|
        # Now reconstruct paths
        # A left node (u,v) matched to right node (v,w) means edge (u,v) is followed by (v,w)
        # Find path starts: left nodes that are not matched as a right node
        # i.e., edges (u,v) where no edge (x,u) with x != v is matched to (u,v)
        # Actually: (u,v) is a path start if there's no matched edge ending at u that transitions to (u,v)
        # (u,v) is a path end if there's no matched transition from (u,v)

        # Let me think more carefully.
        # Left node i represents edge e_i = (u_i, v_i)
        # If left i is matched to right j, it means e_i is followed by e_j in a path.
        # So e_i -> e_j means v_i = u_j (the end of e_i is the start of e_j).
        # A path starts at an edge e_i that is NOT the target of any matched transition.
        # i.e., no left node k is matched to right i.
        # A path ends at an edge e_i that is NOT the source of any matched transition.
        # i.e., left i is not matched (pair_u[i] == -1).

        # Find starts: edges that are not matched as right
        matched_as_right = set()
        for u in range(n_left):
            if pair_u[u] != -1:
                matched_as_right.add(pair_u[u])

        starts = []
        for i in range(n_left):
            if i not in matched_as_right:
                starts.append(i)

        # Follow chains
        paths = []
        visited = [False] * n_left
        for s in starts:
            if visited[s]:
                continue
            path = [left_nodes[s]]
            visited[s] = True
            cur = s
            while pair_u[cur] != -1:
                nxt = pair_u[cur]
                if visited[nxt]:
                    break
                path.append(left_nodes[nxt])
                visited[nxt] = True
                cur = nxt
            paths.append(path)

        # Convert edge paths to vertex paths
        vertex_paths = []
        for ep in paths:
            if len(ep) == 1:
                vertex_paths.append(list(ep[0]))
            else:
                vp = [ep[0][0]]
                for i in range(len(ep)):
                    u, v = ep[i]
                    if i == 0:
                        vp.append(v)
                    else:
                        # Ensure continuity
                        if vp[-1] == u:
                            vp.append(v)
                        else:
                            vp.append(u)
                            vp.append(v)
                # Deduplicate consecutive
                deduped = [vp[0]]
                for v in vp[1:]:
                    if v != deduped[-1]:
                        deduped.append(v)
                vertex_paths.append(deduped)

        return vertex_paths

    def split_and_cover(edges, deadline):
        """
        Split high-degree vertices to reduce the problem, solve min path cover on subgraphs,
        then merge.
        """
        if not edges:
            return []
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = build_adj(edges)
        degree = {v: len(adj[v]) for v in adj}

        # For vertices with high degree, we can't avoid them being in multiple paths.
        # The key insight: if we can partition edges into subgraphs where each vertex
        # has degree <= 2 within its subgraph, then each subgraph is a collection of
        # paths and cycles.

        # Strategy: use edge coloring / greedy partitioning
        # Assign each edge to one of k "layers" such that in each layer, max degree <= 2
        # This is equivalent to edge coloring with k colors where k >= max_degree
        # But we want to MINIMIZE total paths, not just layers.

        # Better approach: directly use the matching-based min path cover.
        # The formula: min paths = |E| - max matching in the line graph bipartite.
        # This is exact! Let's just use it directly.
        paths = min_path_cover_matching(edge_set)
        return paths

    def local_search(paths, edges, deadline):
        """
        Try to reduce path count by:
        1. Merging paths that share endpoints
        2. Re-routing: if two paths have edges that could be swapped to form longer paths
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(current)
            # Try all pairs for merging
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = current[i], current[j]
                    # Try p1 + p2
                    if p1[-1] == p2[0]:
                        merged = p1 + p2[1:]
                        if _is_valid_path(merged, edge_set, current, i, j):
                            current = [p for k, p in enumerate(current) if k != i and k != j] + [merged]
                            improved = True
                            break
                    if p2[-1] == p1[0]:
                        merged = p2 + p1[1:]
                        if _is_valid_path(merged, edge_set, current, i, j):
                            current = [p for k, p in enumerate(current) if k != i and k != j] + [merged]
                            improved = True
                            break
                    # Try p1 reversed + p2
                    if p1[0] == p2[0]:
                        merged = list(reversed(p1)) + p2[1:]
                        if _is_valid_path(merged, edge_set, current, i, j):
                            current = [p for k, p in enumerate(current) if k != i and k != j] + [merged]
                            improved = True
                            break
                    if p1[-1] == p2[-1]:
                        merged = p1 + list(reversed(p2))[1:]
                        if _is_valid_path(merged, edge_set, current, i, j):
                            current = [p for k, p in enumerate(current) if k != i and k != j] + [merged]
                            improved = True
                            break
                if improved:
                    break
        return current

    def _is_valid_path(merged, edge_set, current, i, j):
        """Check if merged is a simple path using only edges from current[i] and current[j]."""
        if len(merged) != len(set(merged)):
            return False
        # Get edges used by path i and j
        allowed = set()
        for p in (current[i], current[j]):
            for k in range(len(p) - 1):
                allowed.add(tuple(sorted((p[k], p[k+1]))))
        for k in range(len(merged) - 1):
            e = tuple(sorted((merged[k], merged[k+1])))
            if e not in allowed:
                return False
        return True

    def verify_and_fix(paths, edges):
        """Ensure validity and completeness."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        valid_paths = []
        for p in paths:
            if len(p) < 2:
                continue
            # Check simple
            if len(p) != len(set(p)):
                seen = set()
                fixed = []
                for v in p:
                    if v in seen:
                        break
                    seen.add(v)
                    fixed.append(v)
                p = fixed
                if len(p) < 2:
                    continue
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set or e in used:
                    ok = False
                    break
            if ok:
                for i in range(len(p) - 1):
                    used.add(tuple(sorted((p[i], p[i+1]))))
                valid_paths.append(p)
        # Add missing edges
        for e in edge_set:
            if e not in used:
                valid_paths.append([e[0], e[1]])
                used.add(e)
        return valid_paths

    def decompose_one(edges, deadline):
        if not edges:
            return []
        best = None
        best_count = len(edges)

        # Method 1: Exact min path cover via matching
        if time.time() < deadline - 10:
            paths = min_path_cover_matching(set(tuple(sorted(e)) for e in edges))
            paths = verify_and_fix(paths, edges)
            paths = local_search(paths, edges, deadline - 5)
            paths = verify_and_fix(paths, edges)
            if len(paths) < best_count:
                best_count = len(paths)
                best = paths

        # Method 2: Greedy with multiple strategies
        for seed in range(30):
            if time.time() > deadline - 2:
                break
            rng = random.Random(seed * 1000 + 42)
            remaining = set(tuple(sorted(e)) for e in edges)
            adj = build_adj(list(remaining))
            degree = {v: len(adj[v]) for v in adj}
            paths = []
            while remaining:
                if time.time() > deadline - 1:
                    break
                # Pick start vertex: prefer odd degree, then high degree
                odd_verts = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
                if odd_verts:
                    start = rng.choice(odd_verts)
                else:
                    cands = [v for v in adj if adj[v]]
                    if not cands:
                        break
                    # Prefer high degree
                    cands.sort(key=lambda v: -degree.get(v, 0))
                    start = rng.choice(cands[:max(1, len(cands)//3)])

                # Greedy extend with Warnsdorff
                path = [start]
                visited = {start}
                for _ in range(500):
                    cur = path[-1]
                    nbs = [nb for nb in adj[cur] if nb not in visited]
                    if not nbs:
                        break
                    def score(nb):
                        return sum(1 for x in adj[nb] if x not in visited)
                    nbs.sort(key=score)
                    if rng.random() < 0.7:
                        choice = nbs[0]
                    elif rng.random() < 0.9:
                        choice = rng.choice(nbs[:min(3, len(nbs))])
                    else:
                        choice = rng.choice(nbs)
                    path.append(choice)
                    visited.add(choice)

                if len(path) < 2:
                    if adj.get(start):
                        nb = rng.choice(list(adj[start]))
                        path = [start, nb]
                    else:
                        e = next(iter(remaining))
                        path = [e[0], e[1]]

                path_edges = set()
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e in remaining:
                        path_edges.add(e)
                if not path_edges:
                    e = next(iter(remaining))
                    path = [e[0], e[1]]
                    path_edges = {e}

                for e in path_edges:
                    remaining.discard(e)
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
                    degree[u] = degree.get(u, 0) - 1
                    degree[v] = degree.get(v, 0) - 1
                for v in list(adj.keys()):
                    if not adj[v]:
                        del adj[v]
                paths.append(path)

            paths = verify_and_fix(paths, edges)
            paths = local_search(paths, edges, deadline - 1)
            paths = verify_and_fix(paths, edges)
            if len(paths) < best_count:
                best_count = len(paths)
                best = paths

        if best is None:
            return [[u, v] for u, v in edges]
        return best

    results = []
    for idx, spec in enumerate(((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        remaining_time = deadline - time.time()
        per = remaining_time / (3 - idx)
        paths = decompose_one(edges, time.time() + per)
        results.append(paths)

    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])

    return results
