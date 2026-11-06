"""Rewritten program: stronger cycle-decomposition via randomized multi-start longest-cycle search."""
import time
import random
from collections import defaultdict

def construct():
    t_start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    def decompose(edges, n, time_budget):
        """
        Decompose edges into cycles and single edges, minimizing piece count.
        Strategy:
          - Build adjacency.
          - Repeatedly find a (near-)longest cycle via randomized DFS from multiple starts.
          - Remove cycle edges, repeat.
          - Leftover edges become single-edge pieces.
        """
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []

        # Build base adjacency
        base_adj = defaultdict(set)
        for (u, v) in edge_set:
            base_adj[u].add(v)
            base_adj[v].add(u)

        used_edges = set()
        pieces = []

        def get_remaining_adj():
            adj = defaultdict(set)
            for (u, v) in edge_set:
                if (u, v) in used_edges:
                    continue
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_longest_cycle(adj, n, rng, max_tries=30):
            """
            Find a long cycle using randomized DFS. Try multiple random start orders.
            Returns the longest cycle found (list of vertices) or None.
            """
            best_cycle = None
            best_len = 0

            for _ in range(max_tries):
                # Shuffle vertex order for diversity
                vertices = [v for v in range(n) if adj.get(v)]
                if not vertices:
                    break
                rng.shuffle(vertices)

                color = [0] * n  # 0=white, 1=gray, 2=black
                parent = [-1] * n
                found = None

                def dfs(u, start):
                    nonlocal found
                    color[u] = 1
                    for v in sorted(adj[u]):
                        if color[v] == 0:
                            parent[v] = u
                            if dfs(v, start):
                                return True
                        elif color[v] == 1:
                            # Found a cycle
                            if v == start:
                                cyc = [u]
                                cur = u
                                while cur != start:
                                    cur = parent[cur]
                                    if cur == -1:
                                        return False
                                    cyc.append(cur)
                                cyc.reverse()
                            else:
                                cyc = [u]
                                cur = u
                                while cur != v:
                                    cur = parent[cur]
                                    if cur == -1:
                                        return False
                                    cyc.append(cur)
                                cyc.append(v)
                            if len(cyc) >= 3:
                                found = cyc
                                return True
                    color[u] = 2
                    return False

                for start in vertices:
                    if color[start] == 0:
                        if dfs(start, start):
                            break

                if found and len(found) > best_len:
                    best_len = len(found)
                    best_cycle = found

            return best_cycle

        # Main loop: repeatedly extract cycles
        while True:
            if time.time() > time_budget:
                break

            adj = get_remaining_adj()
            if not adj:
                break

            # Check if any edge remains
            has_edge = False
            for (u, v) in edge_set:
                if (u, v) not in used_edges:
                    has_edge = True
                    break
            if not has_edge:
                break

            rng = random.Random(int(time.time() * 1000000) % (2**31))
            cycle = find_longest_cycle(adj, n, rng, max_tries=20)

            if cycle is None:
                # No cycle found; break out
                break

            # Remove edges in the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                used_edges.add(edge)

            pieces.append(cycle)

        # Remaining edges as single-edge pieces
        for e in edge_set:
            if e not in used_edges:
                pieces.append(list(e))

        return pieces

    results = []
    total_deadline = DEADLINE
    # Allocate time roughly proportional to graph size
    sizes = [spec[0] for spec in specs]
    total_size = sum(sizes)

    for i, spec in enumerate(specs):
        n = spec[0]
        edges = instance(*spec)
        # Allocate time proportional to size, but ensure at least 5s each
        remaining_time = total_deadline - (time.time() - t_start)
        if remaining_time <= 0:
            results.append([list(e) for e in edges])
            continue
        # Proportional allocation with minimum
        frac = n / total_size
        allocated = max(5.0, remaining_time * frac)
        allocated = min(allocated, remaining_time)

        result = decompose(edges, n, t_start + allocated)
        results.append(result)

    return results
