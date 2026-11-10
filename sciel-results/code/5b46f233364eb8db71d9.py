"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, time_budget):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        remaining = set(edges)
        pieces = []

        def build_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        adj = build_adj(remaining)

        def find_best_cycle(start, max_len=80):
            """Find a long cycle starting and ending at start using DFS with pruning."""
            if len(remaining) < 3:
                return None

            best_cycle = None

            # DFS to find longest cycle through start
            def dfs(path, visited, current, depth):
                nonlocal best_cycle
                if depth > max_len:
                    return

                neighbors = sorted(adj.get(current, []), key=lambda x: -len(adj.get(x, [])))
                for nb in neighbors:
                    edge = (min(current, nb), max(current, nb))
                    if edge not in remaining:
                        continue

                    if nb == start and len(path) >= 3:
                        cycle = path[:]
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle[:]
                    elif nb not in visited:
                        path.append(nb)
                        visited.add(nb)
                        dfs(path, visited, nb, depth + 1)
                        path.pop()
                        visited.remove(nb)

            dfs([start], {start}, start, 1)
            return best_cycle

        def find_cycle_from_vertex(v):
            """Find a cycle containing vertex v."""
            if len(remaining) < 3:
                return None
            return find_best_cycle(v)

        def greedy_decompose(rem, adj, time_limit):
            """Greedily extract cycles, preferring longer ones."""
            local_pieces = []
            local_remaining = set(rem)
            local_adj = build_adj(local_remaining)

            while local_remaining:
                if time.time() > time_limit:
                    break

                # Find vertices with degree >= 2
                vertex_degrees = defaultdict(int)
                for u, v in local_remaining:
                    vertex_degrees[u] += 1
                    vertex_degrees[v] += 1

                candidates = [v for v in vertex_degrees if vertex_degrees[v] >= 2]
                if not candidates:
                    break

                # Sort by degree descending to find longer cycles
                candidates.sort(key=lambda x: -vertex_degrees[x])

                best_cycle = None
                # Try top candidates
                for start in candidates[:min(len(candidates), 15)]:
                    cycle = find_best_cycle(start)
                    if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                        best_cycle = cycle

                if best_cycle:
                    # Extract this cycle
                    for i in range(len(best_cycle) - 1):
                        u, v = best_cycle[i], best_cycle[i + 1]
                        edge = (min(u, v), max(u, v))
                        local_remaining.discard(edge)
                    local_pieces.append(best_cycle[:-1][:])
                    local_adj = build_adj(local_remaining)
                else:
                    # No more cycles
                    break

            return local_pieces, local_remaining

        # Initial greedy decomposition
        pieces, remaining = greedy_decompose(remaining, adj, time_budget * 0.4)

        # Local search: try to form cycles from remaining single edges
        # by looking for paths of length 2 that can be extended
        remaining_edges = list(remaining)

        # Try to find cycles in the remaining subgraph using a different approach
        # Use a modified approach: find any cycle in remaining
        def find_any_cycle(rem):
            """Find any cycle in the remaining edges using BFS/DFS."""
            if len(rem) < 3:
                return None

            adj_rem = defaultdict(set)
            for u, v in rem:
                adj_rem[u].add(v)
                adj_rem[v].add(u)

            visited = set()

            def dfs_cycle(start):
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    for nb in adj_rem.get(node, []):
                        if nb == start and len(path) >= 3:
                            return path[:]
                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, path + [nb]))
                return None

            for v in sorted(adj_rem.keys()):
                if v in visited:
                    continue
                cycle = dfs_cycle(v)
                if cycle:
                    return cycle

            return None

        # Iteratively find cycles in remaining
        while remaining:
            if time.time() > time_budget * 0.7:
                break

            cycle = find_any_cycle(remaining)
            if cycle:
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
                pieces.append(cycle[:-1][:])
            else:
                break

        # Add remaining edges as singles
        for e in sorted(remaining):
            pieces.append([e[0], e[1]])

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    # Allocate time budget proportional to expected complexity
    # Smaller graphs get less time, larger graphs get more
    time_alloc = [30, 40, 50]  # seconds per graph

    for i, spec in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        budget = min(time_alloc[i], remaining_time * 0.8)
        edges = instance(*spec)
        result = decompose(edges, budget)
        results.append(result)

    return results
