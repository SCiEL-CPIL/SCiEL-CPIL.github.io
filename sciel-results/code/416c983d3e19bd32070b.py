def construct():
    import random
    import time
    import sys
    from collections import defaultdict, deque

    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def shortest_paths(n, adj):
        dist = [[-1] * n for _ in range(n)]
        for start in range(n):
            dist[start][start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[start][v] == -1:
                        dist[start][v] = dist[start][u] + 1
                        queue.append(v)
        return dist

    def hungarian(cost):
        n = len(cost)
        if n == 0:
            return []
        if n % 2 != 0:
            raise ValueError("n must be even")
        
        if n <= 16:
            from functools import lru_cache
            sys.setrecursionlimit(10000)
            
            @lru_cache(maxsize=None)
            def min_match(mask):
                if mask == 0:
                    return 0, []
                i = -1
                for j in range(n):
                    if not (mask & (1 << j)):
                        i = j
                        break
                if i == -1:
                    return 0, []
                best_cost = float('inf')
                best_pairs = None
                for j in range(i + 1, n):
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << i) | (1 << j)
                        c, pairs = min_match(new_mask)
                        total = cost[i][j] + c
                        if total < best_cost:
                            best_cost = total
                            best_pairs = [(i, j)] + pairs
                return best_cost, best_pairs
            
            _, pairs = min_match(0)
            return pairs if pairs else []
        else:
            all_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    all_pairs.append((cost[i][j], i, j))
            all_pairs.sort()
            used = set()
            pairs = []
            for c, i, j in all_pairs:
                if i not in used and j not in used:
                    pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    if len(pairs) == n // 2:
                        break
            return pairs

    def solve_instance(n, p, seed, deadline):
        edges = instance_graph(n, p, seed)
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
        k = len(odd_vertices) // 2
        
        if k == 0:
            # All even: Eulerian circuit
            edge_set = set((u, v) for u, v in edges)
            remaining = set(edge_set)
            paths = []
            visited = set()
            for u, v in edges:
                if u in visited and v in visited:
                    continue
                comp = set()
                queue = deque([u])
                while queue:
                    x = queue.popleft()
                    if x in comp:
                        continue
                    comp.add(x)
                    for y in adj[x]:
                        if y not in comp:
                            queue.append(y)
                
                local_adj = {x: set() for x in comp}
                for u, v in edges:
                    if u in comp and v in comp:
                        local_adj[u].add(v)
                        local_adj[v].add(u)
                
                circuit = []
                start_node = None
                for x in comp:
                    if local_adj[x]:
                        start_node = x
                        break
                if start_node is None:
                    continue
                stack = [start_node]
                while stack:
                    cur = stack[-1]
                    if local_adj[cur]:
                        nxt = next(iter(local_adj[cur]))
                        local_adj[cur].remove(nxt)
                        local_adj[nxt].remove(cur)
                        stack.append(nxt)
                    else:
                        circuit.append(stack.pop())
                circuit.reverse()
                for i in range(len(circuit) - 1):
                    a, b = circuit[i], circuit[i + 1]
                    remaining.discard((min(a, b), max(a, b)))
                visited |= comp
                paths.append(circuit)
            return paths
        
        dist = shortest_paths(n, adj)
        m = len(odd_vertices)
        cost = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i != j:
                    cost[i][j] = dist[odd_vertices[i]][odd_vertices[j]]
        
        pairs = hungarian(cost)
        
        edge_set = set((u, v) for u, v in edges)
        remaining = set(edge_set)
        
        # Find paths for matched pairs
        base_paths = []
        for i, j in pairs:
            u = odd_vertices[i]
            v = odd_vertices[j]
            parent = {u: None}
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if x == v:
                    break
                for y in adj[x]:
                    if y not in parent and (min(x, y), max(x, y)) in remaining:
                        parent[y] = x
                        queue.append(y)
            
            if v in parent:
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                base_paths.append(path)
                for idx in range(len(path) - 1):
                    a, b = path[idx], path[idx + 1]
                    remaining.discard((min(a, b), max(a, b)))
        
        # Now merge Eulerian circuits into base paths
        # Build adjacency for remaining edges
        rem_adj = {i: set() for i in range(n)}
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find Eulerian circuits
        circuits = []
        visited = set()
        for u, v in remaining:
            if u in visited and v in visited:
                continue
            comp = set()
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if x in comp:
                    continue
                comp.add(x)
                for y in rem_adj[x]:
                    if y not in comp:
                        queue.append(y)
            
            local_adj = {x: set() for x in comp}
            for u, v in remaining:
                if u in comp and v in comp:
                    local_adj[u].add(v)
                    local_adj[v].add(u)
            
            circuit = []
            start_node = None
            for x in comp:
                if local_adj[x]:
                    start_node = x
                    break
            if start_node is None:
                continue
            stack = [start_node]
            while stack:
                cur = stack[-1]
                if local_adj[cur]:
                    nxt = next(iter(local_adj[cur]))
                    local_adj[cur].remove(nxt)
                    local_adj[nxt].remove(cur)
                    stack.append(nxt)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            circuits.append(circuit)
            visited |= comp
        
        # Merge circuits into base paths
        # For each circuit, find a base path that shares a vertex with it, and insert
        final_paths = [list(p) for p in base_paths]
        
        for circuit in circuits:
            # Find a base path that shares a vertex
            merged = False
            for idx, path in enumerate(final_paths):
                path_set = set(path)
                common = path_set & set(circuit)
                if common:
                    # Insert circuit into path at a common vertex
                    # Find position of first common vertex in path
                    insert_pos = -1
                    for i, v in enumerate(path):
                        if v in circuit:
                            insert_pos = i
                            break
                    if insert_pos >= 0:
                        # Insert circuit starting from the common vertex
                        # Rotate circuit so it starts at the common vertex
                        circ_set = set(circuit)
                        common_v = None
                        for v in path:
                            if v in circ_set:
                                common_v = v
                                break
                        if common_v is not None:
                            # Find common_v in circuit
                            circ_idx = circuit.index(common_v)
                            rotated = circuit[circ_idx:] + circuit[:circ_idx]
                            # rotated starts with common_v
                            # Insert rotated[1:] into path after insert_pos
                            final_paths[idx] = path[:insert_pos+1] + rotated[1:] + path[insert_pos+1:]
                            merged = True
                            break
            if not merged:
                # Should not happen if graph is connected
                final_paths.append(circuit)
        
        return final_paths

    start_time = time.time()
    deadline = start_time + 140

    res1 = solve_instance(20, 0.3, 1, deadline)
    if time.time() > deadline:
        return [res1, [], []]
    res2 = solve_instance(25, 0.3, 2, deadline)
    if time.time() > deadline:
        return [res1, res2, []]
    res3 = solve_instance(30, 0.25, 3, deadline)
    return [res1, res2, res3]
