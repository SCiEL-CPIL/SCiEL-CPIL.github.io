import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        edge_set = set(edges)
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == edge_set

    def greedy_random(edges, seed):
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = []
        while remaining:
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            changed = True
            while changed:
                changed = False
                for end in [0, 1]:
                    if end == 0:
                        current = path[0]
                        cands = [nb for nb in adj[current]
                                  if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if cands:
                            nb = rng.choice(cands)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        cands = [nb for nb in adj[current]
                                  if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if cands:
                            nb = rng.choice(cands)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                if not changed:
                    break
            paths.append(path)
        return paths

    def greedy_target(edges, k, seed_val):
        """Try to cover all edges with exactly k simple paths."""
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None
        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = [[] for _ in range(k)]
        edge_list = list(remaining)
        rng.shuffle(edge_list)
        start_edges = edge_list[:k]
        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]
        changed = True
        while changed:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)
            for i in idxs:
                if not paths[i]:
                    continue
                back = paths[i][-1]
                cands = [nb for nb in adj[back] if nb not in paths[i]
                         and ((back, nb) in remaining or (nb, back) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue
                front = paths[i][0]
                cands = [nb for nb in adj[front] if nb not in paths[i]
                         and ((front, nb) in remaining or (nb, front) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue
        if remaining:
            return None
        return paths

    def max_matching_bipartite(left_nodes, right_nodes, adjacency):
        if not left_nodes or not right_nodes:
            return []
        left_index = {node: i for i, node in enumerate(left_nodes)}
        right_index = {node: i for i, node in enumerate(right_nodes)}
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        adj = [[] for _ in range(n_left)]
        for u in left_nodes:
            ui = left_index[u]
            for v in adjacency.get(u, set()):
                if v in right_index:
                    adj[ui].append(right_index[v])
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = []
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False

        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    dfs(u)
        matching = []
        for u in range(n_left):
            if match_left[u] != -1:
                matching.append((left_nodes[u], right_nodes[match_left[u]]))
        return matching

    def merge_endpoints_v2(paths, edges, adj, used_edges, max_iter=100):
        """Improved endpoint merging: try all pairs, not just parity-split."""
        if not paths or len(paths) <= 1:
            return [list(p) for p in paths]
        current = [list(p) for p in paths]
        current_used = set(used_edges)
        edge_set = set(edges)

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if len(current) <= 1:
                break
            # Build list of all endpoints
            endpoints = []
            for i, path in enumerate(current):
                if len(path) >= 2:
                    endpoints.append((i, 0, path[0]))
                    endpoints.append((i, 1, path[-1]))

            # Try to find any valid merge
            merged_any = False
            n_ep = len(endpoints)
            for a in range(n_ep):
                for b in range(a + 1, n_ep):
                    i1, pos1, v1 = endpoints[a]
                    i2, pos2, v2 = endpoints[b]
                    if i1 == i2:
                        continue
                    e = tuple(sorted((v1, v2)))
                    if e not in edge_set or e in current_used:
                        continue
                    # Check simplicity: no shared vertices
                    if set(current[i1]) & set(current[i2]):
                        continue
                    # Build merged path
                    p1 = current[i1]
                    p2 = current[i2]
                    q1 = list(reversed(p1)) if pos1 == 1 else list(p1)
                    q2 = list(reversed(p2)) if pos2 == 1 else list(p2)
                    if q1[-1] != v1 or q2[0] != v2:
                        continue
                    merged = q1 + q2[1:]
                    if len(set(merged)) != len(merged):
                        continue
                    # Validate all edges
                    ok = True
                    for j in range(len(merged) - 1):
                        ej = tuple(sorted((merged[j], merged[j+1])))
                        if ej not in edge_set:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Apply merge
                    new_current = [list(p) for p in current]
                    new_current[i1] = merged
                    del new_current[i2]
                    current_used.add(e)
                    current = new_current
                    merged_any = True
                    break
                if merged_any:
                    break
            if not merged_any:
                break
        return current

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2

        best_paths = None
        best_count = float('inf')
        max_attempts = 500

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            seed_val = seed * 10000 + attempt

            if attempt % 3 == 0:
                init = greedy_target(edges, lb, seed_val)
                if init is None:
                    init = greedy_target(edges, lb + 1, seed_val)
                if init is None:
                    init = greedy_random(edges, seed_val)
            elif attempt % 3 == 1:
                init = greedy_random(edges, seed_val)
            else:
                init = greedy_target(edges, lb + 1, seed_val)
                if init is None:
                    init = greedy_random(edges, seed_val)

            if not init:
                continue

            used_edges = set()
            for path in init:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    used_edges.add(tuple(sorted((u, v))))

            # Merge endpoints with improved algorithm
            merged = merge_endpoints_v2(init, edges, adj, used_edges)

            if validate_paths(merged, edges):
                if len(merged) < best_count:
                    best_count = len(merged)
                    best_paths = [list(p) for p in merged]
                    if best_count <= lb:
                        break

        if best_paths is None:
            for i in range(100):
                if time.time() > deadline - 5:
                    break
                paths = greedy_random(edges, seed * 10000 + 50000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = [list(p) for p in paths]

        if best_paths is None:
            best_paths = [[u, v] for u, v in edges]

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    for spec in specs:
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
