"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    edges = P['edges']

    # Build adjacency and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es

    def find_ham_cycle(used_edges, start=0, time_budget=None):
        """Find a Hamiltonian cycle using backtracking with pruning."""
        if time_budget is None:
            time_budget = deadline - time.time() - 3

        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    rem_adj[u].add(v)

        # Check minimum degree >= 2
        for u in range(n):
            if len(rem_adj[u]) < 2:
                return None

        # Check connectivity of remaining graph
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in rem_adj[node]:
                if nb not in visited:
                    stack.append(nb)
        if len(visited) != n:
            return None

        # Backtracking: fix start vertex, build path
        path = [start]
        visited_set = {start}
        edges_used_local = set()

        # Iterative deepening with MRV heuristic
        def dfs(current, path, visited_set, edges_used_local):
            if time.time() > deadline - 3:
                return None

            if len(path) == n:
                last = path[-1]
                e = (min(last, start), max(last, start))
                if e not in edges_used_local and e in edge_set:
                    return list(path)
                return None

            current = path[-1]
            # MRV: sort neighbors by fewest remaining options
            candidates = []
            for nb in rem_adj[current]:
                if nb not in visited_set:
                    e = (min(current, nb), max(current, nb))
                    if e not in edges_used_local and e in edge_set:
                        candidates.append((len(rem_adj[nb]), nb))

            candidates.sort()

            for _, nb in candidates:
                e = (min(current, nb), max(current, nb))
                if e in edges_used_local:
                    continue

                new_visited = visited_set | {nb}
                new_edges = edges_used_local | {e}

                # Pruning: check that unvisited vertices still have degree >= 2
                feasible = True
                for v in range(n):
                    if v in new_visited:
                        continue
                    avail = 0
                    for nb2 in rem_adj[v]:
                        e2 = (min(v, nb2), max(v, nb2))
                        if e2 not in new_edges:
                            avail += 1
                    if avail < 2:
                        feasible = False
                        break

                if not feasible:
                    continue

                result = dfs(nb, path + [nb], new_visited, new_edges)
                if result is not None:
                    return result

            return None

        return dfs(start, path, visited_set, edges_used_local)

    # Main search: greedily find edge-disjoint Hamiltonian cycles
    best_cycles = []

    # Try multiple random orderings for diversity
    random.seed(42)

    # Try different starting vertices to break symmetry
    for start_v in range(n):
        if time.time() > deadline - 15:
            break

        for attempt in range(5):
            if time.time() > deadline - 10:
                break

            used_edges = set()
            cycles = []

            for _ in range(10):
                if time.time() > deadline - 5:
                    break

                cycle = find_ham_cycle(used_edges, start=start_v)
                if cycle is None:
                    break

                ce = get_cycle_edges(cycle)
                used_edges |= ce
                cycles.append(cycle)

            if len(cycles) > len(best_cycles):
                best_cycles = list(cycles)

            if len(best_cycles) >= 3:
                break

        if len(best_cycles) >= 3:
            break

    # If we found cycles, return them
    if best_cycles:
        # Ensure proper types
        result = []
        for c in best_cycles:
            result.append([int(v) for v in c])
        return result

    # Fallback: find any valid cycle
    # Try a simple 3-cycle
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return [[int(u), int(v), int(w)]]

    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
