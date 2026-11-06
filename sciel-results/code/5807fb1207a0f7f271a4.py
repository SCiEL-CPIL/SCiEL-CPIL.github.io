"""Path decomposition of graphs using greedy longest-path with Warnsdorff heuristic."""
import time
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into minimum number of simple paths."""
        if not edges:
            return []

        # Build adjacency with edge tracking
        adj = defaultdict(list)  # vertex -> list of (neighbor, edge)
        edge_set = set(edges)
        for u, v in edges:
            adj[u].append((v, (u, v)))
            adj[v].append((u, (u, v)))

        # Track used edges
        used_edges = set()
        paths = []

        # Compute degrees
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        # Sort vertices by degree (prefer starting at odd-degree or low-degree vertices)
        # Odd-degree vertices must be endpoints of some path
        odd_vertices = [v for v in degree if degree[v] % 2 == 1]
        even_vertices = [v for v in degree if degree[v] % 2 == 0]

        # Prioritize odd-degree vertices first, then by degree
        start_candidates = sorted(odd_vertices, key=lambda v: degree[v])
        start_candidates += sorted(even_vertices, key=lambda v: degree[v])

        def find_longest_path(start, remaining_edges):
            """Find a long simple path starting from 'start' using Warnsdorff heuristic."""
            if not remaining_edges:
                return [start]

            # Build adjacency for remaining edges
            rem_adj = defaultdict(list)
            for u, v in remaining_edges:
                rem_adj[u].append(v)
                rem_adj[v].append(u)

            path = [start]
            visited = {start}

            while True:
                current = path[-1]
                # Find unvisited neighbors
                neighbors = [n for n in rem_adj.get(current, []) if n not in visited]
                if not neighbors:
                    break

                # Warnsdorff: choose neighbor with fewest onward options
                def onward_count(n):
                    return len([m for m in rem_adj.get(n, []) if m not in visited and m != path[-2] if len(path) > 1])

                neighbors.sort(key=lambda n: (onward_count(n), degree.get(n, 0)))

                next_vertex = neighbors[0]
                path.append(next_vertex)
                visited.add(next_vertex)

            return path

        # Greedy: repeatedly find longest path from best start vertex
        remaining = set(edges)

        while remaining:
            if time.time() - start_time > DEADLINE:
                # Fallback: each remaining edge as its own path
                for e in remaining:
                    paths.append(list(e))
                break

            # Find the start vertex that gives the longest path
            best_path = None
            best_start = None

            # Try odd-degree vertices first, then others
            candidates = list(remaining_vertices(remaining, adj, used_edges))
            # Prioritize odd-degree
            candidates.sort(key=lambda v: (degree[v] % 2 == 0, degree[v]))

            # Limit candidates for speed
            for start in candidates[:50]:
                path = find_longest_path(start, remaining)
                if best_path is None or len(path) > len(best_path):
                    best_path = path
                    best_start = start

                if len(path) >= len(remaining):  # Can't do better
                    break

            if best_path is None or len(best_path) < 2:
                # Single vertex or no path found, take one edge
                e = next(iter(remaining))
                best_path = list(e)
            else:
                # Convert path to edges and remove them
                path_edges = set()
                for i in range(len(best_path) - 1):
                    e = tuple(sorted((best_path[i], best_path[i+1])))
                    if e in remaining:
                        path_edges.add(e)
                    else:
                        # Edge not in remaining, truncate
                        best_path = best_path[:i+1]
                        break
                remaining -= path_edges

            paths.append(best_path)

        return paths

    def remaining_vertices(remaining, adj, used_edges):
        """Get vertices that still have unused edges incident to them."""
        verts = set()
        for u, v in remaining:
            verts.add(u)
            verts.add(v)
        return verts

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Emergency fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
