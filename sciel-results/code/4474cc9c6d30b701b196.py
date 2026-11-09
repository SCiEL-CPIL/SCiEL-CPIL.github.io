import time
import random
from collections import defaultdict

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    n = 21
    edges = [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]

    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        key = (min(a, b), max(a, b))
        edge_set.add(key)
        adj[a].add(b)
        adj[b].add(a)

    def find_hamiltonian_cycle(available, start, max_time=5.0):
        """Find a Hamiltonian cycle using backtracking with strong pruning."""
        t0 = time.time()
        path = [start]
        path_set = {start}
        used_edges = set()

        # Precompute adjacency as sorted lists for speed
        adj_list = [list(adj[i]) for i in range(n)]

        def backtrack():
            if time.time() - t0 > max_time:
                return None
            if len(path) == n:
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in available and key not in used_edges:
                    return list(path)
                return None

            last = path[-1]
            # Get candidate neighbors: not in path, edge available
            candidates = []
            for v in adj_list[last]:
                if v in path_set:
                    continue
                key = (min(last, v), max(last, v))
                if key not in available or key in used_edges:
                    continue
                candidates.append(v)

            if not candidates:
                return None

            # Warnsdorff heuristic: prefer neighbors with fewer onward options
            def onward_count(v):
                cnt = 0
                for u in adj_list[v]:
                    if u in path_set:
                        continue
                    k = (min(v, u), max(v, u))
                    if k in available and k not in used_edges:
                        cnt += 1
                return cnt

            candidates.sort(key=lambda v: (onward_count(v), v))

            for nb in candidates:
                key = (min(last, nb), max(last, nb))
                used_edges.add(key)
                path.append(nb)
                path_set.add(nb)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                path_set.discard(nb)
                used_edges.discard(key)
            return None

        return backtrack()

    def try_find_cycle(available, start, max_time):
        """Try to find a Hamiltonian cycle with the given start."""
        return find_hamiltonian_cycle(available, start, max_time)

    # Strategy: iteratively find edge-disjoint Hamiltonian cycles
    # Use constraint propagation to prune before backtracking
    cycles = []
    available = set(edge_set)

    # Order of starting vertices to try (based on graph structure)
    # Try multiple orderings to increase success probability
    orderings = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [10, 0, 20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14, 7, 13, 8, 12, 9, 11],
    ]

    max_cycles_target = 10
    total_time_budget = deadline - t_start

    for ordering in orderings:
        if len(cycles) >= max_cycles_target:
            break
        if time.time() > deadline - 10:
            break

        # Time per cycle search decreases as we find more
        time_per_cycle = max(3.0, (deadline - time.time()) / max(1, max_cycles_target - len(cycles)) - 2)

        for start in ordering:
            if len(cycles) >= max_cycles_target:
                break
            if time.time() > deadline - 10:
                break

            cycle = try_find_cycle(available, start, time_per_cycle)
            if cycle is not None:
                cycles.append(cycle)
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    key = (min(a, b), max(a, b))
                    available.discard(key)
            else:
                # If we can't find a cycle with this start, try next
                continue

    # If we only found few cycles, try randomized restarts with different edge removals
    if len(cycles) < 4:
        # Try a different approach: find cycles in a more structured way
        # Reset and try with different strategies
        available = set(edge_set)
        cycles2 = []

        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            if len(cycles2) >= max_cycles_target:
                break

            # Randomize the order of edge removal consideration
            remaining_edges = list(available)
            random.shuffle(remaining_edges)

            for start in range(n):
                if len(cycles2) >= max_cycles_target:
                    break
                if time.time() > deadline - 15:
                    break

                cycle = find_hamiltonian_cycle(available, start, 2.0)
                if cycle is not None:
                    cycles2.append(cycle)
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        key = (min(a, b), max(a, b))
                        available.discard(key)

        if len(cycles2) > len(cycles):
            cycles = cycles2

    # Ensure at least one cycle
    if len(cycles) == 0:
        available = set(edge_set)
        for s in range(n):
            cycle = find_hamiltonian_cycle(available, s, 10.0)
            if cycle is not None:
                cycles = [cycle]
                break

    cycles = cycles[:10]

    # Convert to list of lists of Python ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])

    return result
