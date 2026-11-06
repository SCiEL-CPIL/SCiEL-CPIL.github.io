"""Program that decomposes graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        remaining = set(edge_set)
        pieces = []

        def remove_cycle(cycle):
            """Remove all edges of a cycle from remaining set."""
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)

        def find_cycle_from_start(start, max_depth=30):
            """Find a cycle starting from start using DFS with backtracking."""
            visited = [False] * n
            parent = [-1] * n
            cycle = [None]

            def dfs(node, depth):
                if cycle[0] is not None:
                    return True
                if depth > max_depth:
                    return False
                visited[node] = True
                for neighbor in sorted(adj[node]):
                    if (min(node, neighbor), max(node, neighbor)) not in remaining:
                        continue
                    if neighbor == start and depth >= 2:
                        # Found a cycle
                        c = [start]
                        cur = node
                        while cur != start:
                            c.append(cur)
                            cur = parent[cur]
                        cycle[0] = c
                        return True
                    if not visited[neighbor]:
                        parent[neighbor] = node
                        if dfs(neighbor, depth + 1):
                            return True
                visited[node] = False
                return False

            dfs(start, 0)
            return cycle[0]

        def find_longest_cycle_randomized():
            """Try to find a long cycle using randomized DFS from multiple starts."""
            best = None
            rng = random.Random(123)
            
            # Try several random starts
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            for _ in range(min(len(candidates), 15)):
                start = rng.choice(candidates)
                if start not in candidates:
                    continue
                # Shuffle neighbor order for diversity
                adj_copy = {v: sorted(adj[v]) for v in range(n)}
                for v in adj_copy:
                    rng.shuffle(adj_copy[v])
                
                # Temporarily use shuffled adjacency
                old_adj = adj
                adj = adj_copy
                
                visited = [False] * n
                parent = [-1] * n
                cycle = [None]
                
                def dfs(node, depth):
                    if cycle[0] is not None:
                        return True
                    if depth > 25:
                        return False
                    visited[node] = True
                    for neighbor in adj[node]:
                        if (min(node, neighbor), max(node, neighbor)) not in remaining:
                            continue
                        if neighbor == start and depth >= 2:
                            c = [start]
                            cur = node
                            while cur != start:
                                c.append(cur)
                                cur = parent[cur]
                            cycle[0] = c
                            return True
                        if not visited[neighbor]:
                            parent[neighbor] = node
                            if dfs(neighbor, depth + 1):
                                return True
                    visited[node] = False
                    return False
                
                dfs(start, 0)
                adj = old_adj
                
                if cycle[0] is not None:
                    if best is None or len(cycle[0]) > len(best):
                        best = cycle[0]
                        if len(best) >= 8:
                            return best
            
            return best

        def find_cycle_greedy():
            """Find a cycle by greedily following unused edges."""
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                
                # Try to build a cycle from start
                path = [start]
                visited = {start}
                current = start
                
                for _ in range(20):
                    # Find an unvisited neighbor with available edges
                    next_node = None
                    for neighbor in sorted(adj[current]):
                        if (min(current, neighbor), max(current, neighbor)) not in remaining:
                            continue
                        if neighbor == start and len(path) >= 3:
                            next_node = start
                            break
                        if neighbor not in visited:
                            next_node = neighbor
                            break
                    
                    if next_node is None:
                        break
                    if next_node == start:
                        return path
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
            
            return None

        def find_all_cycles_greedy():
            """Try to extract multiple disjoint cycles at once."""
            cycles = []
            temp_remaining = set(remaining)
            
            # Try to find cycles one by one, ensuring they're edge-disjoint
            for _ in range(10):
                if time.time() > deadline - 15:
                    break
                
                # Find a cycle in the current remaining edges
                found = None
                for start in range(n):
                    if len(adj[start]) < 2:
                        continue
                    
                    # DFS to find a cycle
                    visited = [False] * n
                    parent = [-1] * n
                    cycle = [None]
                    
                    def dfs(node, depth):
                        if cycle[0] is not None:
                            return True
                        if depth > 20:
                            return False
                        visited[node] = True
                        for neighbor in sorted(adj[node]):
                            e = (min(node, neighbor), max(node, neighbor))
                            if e not in temp_remaining:
                                continue
                            if neighbor == start and depth >= 2:
                                c = [start]
                                cur = node
                                while cur != start:
                                    c.append(cur)
                                    cur = parent[cur]
                                cycle[0] = c
                                return True
                            if not visited[neighbor]:
                                parent[neighbor] = node
                                if dfs(neighbor, depth + 1):
                                    return True
                        visited[node] = False
                        return False
                    
                    dfs(start, 0)
                    
                    if cycle[0] is not None:
                        found = cycle[0]
                        break
                
                if found is None:
                    break
                
                # Remove edges of this cycle from temp_remaining
                for i in range(len(found)):
                    u, v = found[i], found[(i + 1) % len(found)]
                    e = (min(u, v), max(u, v))
                    temp_remaining.discard(e)
                
                cycles.append(found)
            
            return cycles

        # Main decomposition loop
        max_iterations = 200
        iter_count = 0
        while iter_count < max_iterations:
            iter_count += 1
            if time.time() > deadline - 10:
                break
            
            if not remaining:
                break
            
            # Strategy 1: Try to find multiple cycles at once
            multi_cycles = find_all_cycles_greedy()
            if multi_cycles and len(multi_cycles) >= 2:
                for cycle in multi_cycles:
                    remove_cycle(cycle)
                    pieces.append(cycle)
                continue
            
            # Strategy 2: Try to find a long cycle
            cycle = find_longest_cycle_randomized()
            
            # Strategy 3: Fallback to simple cycle finding
            if cycle is None:
                for start in range(n):
                    if len(adj[start]) < 2:
                        continue
                    cycle = find_cycle_from_start(start)
                    if cycle is not None:
                        break
            
            # Strategy 4: Greedy cycle
            if cycle is None:
                cycle = find_cycle_greedy()
            
            if cycle is None:
                break
            
            remove_cycle(cycle)
            pieces.append(cycle)

        # Remaining edges become individual pieces
        for u, v in sorted(remaining):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
