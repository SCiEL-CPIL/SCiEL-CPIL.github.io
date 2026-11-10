import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85

    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

    n = P['n']
    edges = P['edges']

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

    def find_ham_cycle(used_edges, start=0):
        if time.time() > deadline - 3:
            return None

        # Build remaining adjacency and degree
        rem_adj = [set() for _ in range(n)]
        rem_deg = [0] * n
        for u in range(n):
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    rem_adj[u].add(v)
                    rem_deg[u] += 1

        # Check min degree >= 2
        for u in range(n):
            if rem_deg[u] < 2:
                return None

        # Check connectivity via BFS
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

        # The start vertex must have degree >= 2 in remaining graph
        # and one of its edges must be usable as the closing edge

        path = [start]
        visited_set = {start}
        edges_used_local = set()

        def dfs(current, depth):
            if time.time() > deadline - 3:
                return None

            if depth == n - 1:
                last = path[-1]
                e = (min(last, start), max(last, start))
                if e not in edges_used_local and e in edge_set:
                    return list(path)
                return None

            current = path[-1]
            # Get candidates: unvisited neighbors
            candidates = []
            for nb in rem_adj[current]:
                if nb not in visited_set:
                    e = (min(current, nb), max(current, nb))
                    if e not in edges_used_local and e in edge_set:
                        # Warnsdorff: prefer neighbors with fewer remaining options
                        # Count how many unvisited neighbors nb has (excluding current)
                        opts = 0
                        for nn in rem_adj[nb]:
                            if nn not in visited_set and nn != current:
                                opts += 1
                        # Also check if nb is the last vertex to be visited
                        # (i.e., only nb and current are unvisited among nb's neighbors)
                        candidates.append((opts, nb))

            candidates.sort()

            for _, nb in candidates:
                e = (min(current, nb), max(current, nb))
                if e in edges_used_local:
                    continue

                # Pruning: after adding nb, check if any unvisited vertex
                # would have fewer than 2 available edges
                visited_set.add(nb)
                edges_used_local.add(e)
                path.append(nb)

                # Quick prune: check degrees of unvisited vertices
                valid = True
                if depth < n - 2:  # not the last step
                    for v in range(n):
                        if v in visited_set:
                            continue
                        # Count available edges from v to unvisited vertices
                        avail = 0
                        for nn in rem_adj[v]:
                            if nn not in visited_set:
                                avail += 1
                        if avail < 2:
                            valid = False
                            break

                if valid:
                    result = dfs(nb, depth + 1)
                    if result is not None:
                        return result
                else:
                    pass  # skip this candidate

                # Backtrack
                path.pop()
                visited_set.remove(nb)
                edges_used_local.remove(e)

            return None

        return dfs(start, 1)

    best_cycles = []

    # Try multiple strategies
    strategies = [
        # (start_vertex, neighbor_order_hint)
    ]

    # First try: start from vertex 0
    for start_v in range(n):
        if time.time() > deadline - 10:
            break

        used_edges = set()
        cycles = []

        for _ in range(3):
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

    if best_cycles:
        result = []
        for c in best_cycles:
            result.append([int(v) for v in c])
        return result

    # Fallback: try a few more starts
    for start_v in [0, 1, 2, 5, 10]:
        if time.time() > deadline - 5:
            break
        used_edges = set()
        cycles = []
        for _ in range(3):
            if time.time() > deadline - 3:
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

    if best_cycles:
        result = []
        for c in best_cycles:
            result.append([int(v) for v in c])
        return result

    # Final fallback
    return [[0, 1, 2]]
