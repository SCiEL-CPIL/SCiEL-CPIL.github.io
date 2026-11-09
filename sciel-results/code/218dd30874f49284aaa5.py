import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve(edges, n, time_budget):
        """Find minimum edge-disjoint path cover. Returns list of paths."""
        if not edges:
            return []

        # Build adjacency and neighbor-edge lookup
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for v in adj:
            adj[v].sort()

        # For each vertex, list of (neighbor, edge_tuple) for fast removal check
        nbr_edges = {}
        for v in adj:
            nbr_edges[v] = [(nb, (v, nb) if v < nb else (nb, v)) for nb in adj[v]]

        # Degrees
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        odd = [v for v in range(n) if degree[v] % 2 == 1]
        if not odd:
            # Eulerian: one path (cycle) covers all edges
            # Find Eulerian cycle
            return _eulerian_cycle(edges, adj, n)

        m = len(odd) // 2  # number of paths needed

        # Greedy initial pairing (sorted)
        initial_pairs = [(odd[i], odd[i + 1]) for i in range(0, len(odd), 2)]

        # Try to find a full decomposition with a given pairing
        def try_decompose(pairs, time_deadline):
            """Try to find edge-disjoint simple paths for the given pairing.
            Returns list of paths if successful, None otherwise."""
            remaining = set(edges)
            paths = []
            # Track which edges are used (for fast check)
            # We'll use a set for O(1) lookup

            # Sort pairs by difficulty: pairs with fewer connecting options first
            # Compute "difficulty" = number of edges incident to either endpoint
            def pair_difficulty(p):
                a, b = p
                return degree[a] + degree[b]
            # Try harder pairs first (more constrained)
            sorted_pairs = sorted(pairs, key=lambda p: -pair_difficulty(p))

            # Backtracking: assign paths to pairs
            # We'll use recursive backtracking over pairs
            assignment = [None] * len(sorted_pairs)

            def find_path(a, b, remaining, adj, nbr_edges, n, max_depth=60):
                """Find simple path from a to b using only edges in remaining."""
                if a == b:
                    return [a]
                visited = [False] * n
                visited[a] = True
                path = [a]

                def dfs(node):
                    if node == b:
                        return True
                    if len(path) > max_depth:
                        return False
                    # Try neighbors in order of increasing remaining degree
                    # (dead-end heuristic)
                    nbrs = nbr_edges[node]
                    # Sort by remaining degree of neighbor (ascending = more constrained first)
                    def rem_deg(nb):
                        cnt = 0
                        for nb2, e2 in nbr_edges[nb]:
                            if e2 in remaining and not visited[nb2]:
                                cnt += 1
                        return cnt
                    # Get available neighbors
                    avail = []
                    for nb, e in nbrs:
                        if visited[nb]:
                            continue
                        if e not in remaining:
                            continue
                        avail.append((rem_deg(nb), nb, e))
                    avail.sort(key=lambda x: x[0])
                    for _, nb, e in avail:
                        remaining.discard(e)
                        visited[nb] = True
                        path.append(nb)
                        if dfs(nb):
                            return True
                        path.pop()
                        visited[nb] = False
                        remaining.add(e)
                    return False

                if dfs(a):
                    return list(path)
                return None

            def backtrack(idx):
                if time.time() > time_deadline:
                    return "timeout"
                if idx == len(sorted_pairs):
                    return True
                a, b = sorted_pairs[idx]
                # Try to find a path from a to b
                path = find_path(a, b, remaining, adj, nbr_edges, n)
                if path is None:
                    return False
                # Use this path
                for i in range(len(path) - 1):
                    e = (path[i], path[i+1]) if path[i] < path[i+1] else (path[i+1], path[i])
                    remaining.discard(e)
                assignment[idx] = path
                if backtrack(idx + 1):
                    return True
                # Undo
                for i in range(len(path) - 1):
                    e = (path[i], path[i+1]) if path[i] < path[i+1] else (path[i+1], path[i])
                    remaining.add(e)
                assignment[idx] = None
                return False

            # Try initial pairing
            if backtrack(0):
                if assignment[-1] is not None or len(sorted_pairs) == 0:
                    return [assignment[i] for i in range(len(sorted_pairs))]
            return None

        # First, try the initial (sorted) pairing
        result = try_decompose(initial_pairs, time_limit - 5)
        if result is not None:
            return result

        # Random restarts: try random pairings
        rng = random.Random(42)
        attempts = 0
        while time.time() < time_limit - 3:
            attempts += 1
            # Random pairing
            shuffled_odd = list(odd)
            rng.shuffle(shuffled_odd)
            pairs = [(shuffled_odd[i], shuffled_odd[i+1]) for i in range(0, len(shuffled_odd), 2)]
            result = try_decompose(pairs, time_limit - 3)
            if result is not None:
                return result
            # Every 50 attempts, try a slightly different strategy
            if attempts % 50 == 0:
                # Try greedy pairing: pair vertices that are close
                # (have a short path between them)
                # This is harder to implement quickly, so just keep random
                pass

        # Fallback: greedy decomposition (may not be optimal but valid)
        return _greedy_decompose(edges, adj, nbr_edges, n, time_limit)

    def _eulerian_cycle(edges, adj, n):
        """Find an Eulerian cycle (all vertices even degree)."""
        # Hierholzer's algorithm
        adj_set = defaultdict(set)
        for u, v in edges:
            adj_set[u].add(v)
            adj_set[v].add(u)

        # Use a stack-based approach
        # We need to handle multigraph-like behavior (but here simple graph)
        # Each edge used once
        edge_used = set()
        start = list(adj_set.keys())[0] if adj_set else 0

        stack = [start]
        cycle = []
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            found = None
            for nb in adj_set[v]:
                e = (v, nb) if v < nb else (nb, v)
                if e not in edge_used:
                    found = (nb, e)
                    break
            if found is None:
                cycle.append(stack.pop())
            else:
                nb, e = found
                edge_used.add(e)
                stack.append(nb)

        cycle.reverse()
        # The cycle is a list of vertices; convert to a path (remove the closing vertex if it equals start)
        # Actually for a cycle, the first and last vertex are the same in the closed walk
        # But we want a path, so if it's a cycle, we can return it as a path by removing the last vertex
        if len(cycle) > 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        return [cycle] if cycle else []

    def _greedy_decompose(edges, adj, nbr_edges, n, time_limit):
        """Greedy: repeatedly find longest simple path in remaining graph."""
        remaining = set(edges)
        paths = []
        while remaining:
            if time.time() > time_limit - 2:
                # Emergency: add remaining edges as single-edge paths
                for e in remaining:
                    paths.append(list(e))
                break
            # Find a long path: start from a vertex with high degree, extend greedily
            # Pick start vertex with highest remaining degree
            best_start = None
            best_deg = -1
            for v in range(n):
                deg = sum(1 for nb, e in nbr_edges[v] if e in remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_start = v
            if best_start is None:
                break
            # Extend path greedily from best_start
            path = [best_start]
            visited = {best_start}
            while True:
                current = path[-1]
                # Find neighbor with highest remaining degree
                best_nb = None
                best_nb_deg = -1
                for nb, e in nbr_edges[current]:
                    if nb in visited:
                        continue
                    if e not in remaining:
                        continue
                    deg = sum(1 for nb2, e2 in nbr_edges[nb] if e2 in remaining and nb2 not in visited)
                    if deg > best_nb_deg:
                        best_nb_deg = deg
                        best_nb = nb
                        best_nb_e = e
                if best_nb is None:
                    break
                remaining.discard(best_nb_e)
                visited.add(best_nb)
                path.append(best_nb)
            # Also extend from the other end
            while True:
                current = path[0]
                best_nb = None
                best_nb_deg = -1
                for nb, e in nbr_edges[current]:
                    if nb in visited:
                        continue
                    if e not in remaining:
                        continue
                    deg = sum(1 for nb2, e2 in nbr_edges[nb] if e2 in remaining and nb2 not in visited)
                    if deg > best_nb_deg:
                        best_nb_deg = deg
                        best_nb = nb
                        best_nb_e = e
                if best_nb is None:
                    break
                remaining.discard(best_nb_e)
                visited.add(best_nb)
                path.insert(0, best_nb)
            if len(path) >= 2:
                paths.append(path)
            else:
                # Single vertex, shouldn't happen if remaining is non-empty
                break
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        # Allocate time budget proportionally
        elapsed = time.time() - start_time
        remaining_time = max(1.0, time_limit - elapsed - 2)
        paths = solve(edges, n, remaining_time)
        results.append(paths)

    return results
