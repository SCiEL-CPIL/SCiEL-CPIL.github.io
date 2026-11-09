import time
from collections import defaultdict, deque
import random as _rng_module

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        if time.time() > deadline:
            # Emergency: return all edges as singletons for remaining graphs
            for remaining_spec in specs[specs.index(spec):]:
                rem_edges = instance(*remaining_spec)
                results.append([list(e) for e in rem_edges])
            break

        edges = instance(*spec)
        n = spec[0]

        # Build adjacency
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        pieces = []

        # Phase 1: Repeatedly find longest cycle in the 2-core
        # We'll use a randomized approach: try multiple starting points, 
        # do DFS with backtracking to find longest cycle within time budget
        max_cycle_attempts = 200  # cap on number of cycle-finding iterations
        cycle_iter = 0

        while edge_set and cycle_iter < max_cycle_attempts:
            if time.time() > deadline - 5.0:
                break
            cycle_iter += 1

            # Find the 2-core: iteratively remove degree-0 and degree-1 vertices
            # The 2-core is where all cycles live
            deg = {u: len(adj[u]) for u in range(n)}
            # Only consider vertices that still have edges
            active = set()
            for e in edge_set:
                active.add(e[0])
                active.add(e[1])

            # Peeling to find 2-core
            in_core = set(active)
            queue = deque([u for u in active if len(adj[u]) < 2])
            while queue:
                u = queue.popleft()
                if u not in in_core:
                    continue
                in_core.discard(u)
                for v in list(adj[u]):
                    if v in in_core:
                        # Recheck degree
                        cnt = 0
                        for w in adj[v]:
                            if w in in_core and ((v, w) in edge_set or (w, v) in edge_set):
                                cnt += 1
                        if cnt < 2:
                            queue.append(v)

            if not in_core:
                break

            # Find longest cycle in the 2-core using randomized DFS
            best_cycle = None
            best_len = 0
            core_list = sorted(in_core)
            
            # Try a few random starting points in the core
            rng_local = _rng_module.Random(42 + cycle_iter)
            if len(core_list) > 1:
                starts = rng_local.sample(core_list, min(5, len(core_list)))
            else:
                starts = core_list

            for start in starts:
                if time.time() > deadline - 3.0:
                    break
                
                # DFS to find longest cycle starting and ending at start
                # Use iterative DFS with path tracking
                best_for_start = []
                
                # We do a DFS that tracks the path and looks for cycles back to start
                # To find the LONGEST cycle, we need to explore deeply
                # Use a simple approach: DFS that tries to extend the path
                # and when we can go back to start, record the cycle
                
                path = [start]
                visited_in_path = {start}
                found = []
                
                def dfs_longest(u, path, visited_in_path, depth_limit=30):
                    nonlocal best_for_start
                    if len(path) > len(best_for_start):
                        # Check if we can close the cycle
                        if start in adj[u] and len(path) >= 3:
                            # Check edge exists
                            e = tuple(sorted((u, start)))
                            if e in edge_set:
                                best_for_start = list(path)
                    
                    if depth_limit <= 0:
                        return
                    
                    for v in sorted(adj[u]):
                        e = tuple(sorted((u, v)))
                        if e not in edge_set and (v, u) not in edge_set:
                            continue
                        if v == start and len(path) >= 3:
                            # Found a cycle!
                            if len(path) > len(best_for_start):
                                best_for_start = list(path)
                        elif v not in visited_in_path:
                            visited_in_path.add(v)
                            path.append(v)
                            dfs_longest(v, path, visited_in_path, depth_limit - 1)
                            path.pop()
                            visited_in_path.discard(v)
                
                dfs_longest(start, path, visited_in_path, depth_limit=25)
                
                if len(best_for_start) > best_len:
                    best_len = len(best_for_start)
                    best_cycle = best_for_start

            if best_cycle and len(best_cycle) >= 3:
                # Verify the cycle edges all exist
                valid = True
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set:
                        valid = False
                        break
                if valid:
                    # Remove cycle edges
                    for i in range(len(best_cycle)):
                        u = best_cycle[i]
                        v = best_cycle[(i + 1) % len(best_cycle)]
                        e = tuple(sorted((u, v)))
                        edge_set.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                    pieces.append(best_cycle)
                else:
                    break
            else:
                break

        # Phase 2: Handle remaining edges
        # Try to find any remaining cycles (shorter ones we might have missed)
        remaining_cycle_found = True
        while remaining_cycle_found:
            if time.time() > deadline - 2.0:
                break
            remaining_cycle_found = False
            
            # Simple DFS to find any cycle
            visited = [0] * n
            parent = [-1] * n
            cycle = None
            
            def dfs_any(u):
                nonlocal cycle, remaining_cycle_found
                visited[u] = 1
                for v in sorted(adj[u]):
                    e = tuple(sorted((u, v)))
                    if e not in edge_set:
                        continue
                    if visited[v] == 0:
                        parent[v] = u
                        if dfs_any(v):
                            return True
                    elif visited[v] == 1 and v != parent[u]:
                        # Found cycle
                        cycle_nodes = [u]
                        curr = u
                        while curr != v:
                            curr = parent[curr]
                            cycle_nodes.append(curr)
                        cycle = cycle_nodes[::-1]
                        remaining_cycle_found = True
                        return True
                visited[u] = 2
                return False

            for u in range(n):
                if visited[u] == 0 and adj[u]:
                    if dfs_any(u):
                        break

            if cycle and len(cycle) >= 3:
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set:
                        valid = False
                        break
                if valid:
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        edge_set.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                    pieces.append(cycle)
                else:
                    break

        # Phase 3: All remaining edges are singletons (they form a forest)
        for e in edge_set:
            pieces.append(list(e))

        results.append(pieces)

    # If we broke out early, fill in remaining
    while len(results) < len(specs):
        idx = len(results)
        rem_edges = instance(*specs[idx])
        results.append([list(e) for e in rem_edges])

    return results
