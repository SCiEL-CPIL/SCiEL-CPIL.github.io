import random
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    time_limit = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def bfs_shortest_path(start, end, adj, remaining_edges):
        if start == end:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb in visited:
                    continue
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue
                visited.add(nb)
                parent[nb] = node
                if nb == end:
                    path = [nb]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                queue.append(nb)
        return None

    def greedy_extend(path, adj, remaining_edges, degree):
        path_set = set(path)
        while True:
            current = path[0]
            candidates = [nb for nb in adj[current]
                         if tuple(sorted((current, nb))) in remaining_edges
                         and nb not in path_set]
            if not candidates:
                break
            candidates.sort(key=lambda nb: -degree[nb])
            next_node = candidates[0]
            path.insert(0, next_node)
            path_set.add(next_node)
            remaining_edges.discard(tuple(sorted((current, next_node))))
            degree[current] -= 1
            degree[next_node] -= 1
        while True:
            current = path[-1]
            candidates = [nb for nb in adj[current]
                         if tuple(sorted((current, nb))) in remaining_edges
                         and nb not in path_set]
            if not candidates:
                break
            candidates.sort(key=lambda nb: -degree[nb])
            next_node = candidates[0]
            path.append(next_node)
            path_set.add(next_node)
            remaining_edges.discard(tuple(sorted((current, next_node))))
            degree[current] -= 1
            degree[next_node] -= 1
        return path

    def decompose(edges):
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        degree = defaultdict(int)
        for u, v in remaining_edges:
            degree[u] += 1
            degree[v] += 1

        odd_vertices = sorted([v for v in degree if degree[v] % 2 == 1])
        m = len(odd_vertices) // 2

        best_paths = None
        best_count = float('inf')

        strategies = []
        strategies.append([(odd_vertices[i], odd_vertices[i+1]) for i in range(0, len(odd_vertices), 2)])
        strategies.append([(odd_vertices[i], odd_vertices[len(odd_vertices)-1-i]) for i in range(m)])
        for seed_val in [42, 123, 999, 7, 314]:
            shuffled = odd_vertices[:]
            random.Random(seed_val).shuffle(shuffled)
            strategies.append([(shuffled[i], shuffled[i+1]) for i in range(0, len(shuffled), 2)])

        for pairs in strategies:
            if time.time() - start_time > time_limit - 10:
                break

            trial_remaining = set(remaining_edges)
            trial_degree = dict(degree)
            trial_paths = []
            success = True

            for (a, b) in pairs:
                if time.time() - start_time > time_limit - 5:
                    success = False
                    break
                path = bfs_shortest_path(a, b, adj, trial_remaining)
                if path is None or len(path) < 2:
                    success = False
                    break
                valid = True
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e not in trial_remaining:
                        valid = False
                        break
                if not valid:
                    success = False
                    break
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    trial_remaining.discard(e)
                    trial_degree[path[i]] -= 1
                    trial_degree[path[i+1]] -= 1
                trial_paths.append(path)

            if not success:
                continue

            while trial_remaining:
                if time.time() - start_time > time_limit - 3:
                    break
                best_edge = max(trial_remaining, key=lambda e: trial_degree[e[0]] + trial_degree[e[1]])
                u, v = best_edge
                path = [u, v]
                trial_remaining.discard(best_edge)
                trial_degree[u] -= 1
                trial_degree[v] -= 1
                path = greedy_extend(path, adj, trial_remaining, trial_degree)
                trial_paths.append(path)

            if len(trial_paths) < best_count:
                best_count = len(trial_paths)
                best_paths = trial_paths[:]

        if best_paths is None:
            trial_remaining = set(remaining_edges)
            trial_degree = dict(degree)
            trial_paths = []
            while trial_remaining:
                if time.time() - start_time > time_limit - 2:
                    for e in trial_remaining:
                        trial_paths.append(list(e))
                    break
                best_edge = max(trial_remaining, key=lambda e: trial_degree[e[0]] + trial_degree[e[1]])
                u, v = best_edge
                path = [u, v]
                trial_remaining.discard(best_edge)
                trial_degree[u] -= 1
                trial_degree[v] -= 1
                path = greedy_extend(path, adj, trial_remaining, trial_degree)
                trial_paths.append(path)
            best_paths = trial_paths

        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
