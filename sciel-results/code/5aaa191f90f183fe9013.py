    def decompose(edge_list):
        edge_set = {tuple(sorted(e)) for e in edge_list}
        n = max(max(e) for e in edge_set) + 1 if edge_set else 0
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []

        def remove_cycle(cycle):
            nonlocal edge_set, adj
            m = len(cycle)
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    edge_set.discard(key)
                    adj[a].discard(b)
                    adj[b].discard(a)

        def find_longest_cycle(adj, n):
            best_cycle = []
            # Try to find a cycle starting from each node with degree >= 2
            # To speed up, we can just pick a few random ones or all of them.
            # For n=30, all is fine.
            
            # We use a heuristic: sort neighbors by degree descending
            # This helps find long paths early.
            
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                
                # DFS
                # State: (node, path, visited)
                # We use a stack for iterative DFS to manage backtracking manually
                # However, recursive is easier for "longest" search with backtracking.
                # n=30 is small enough for recursion depth.
                
                def dfs(node, path, visited):
                    nonlocal best_cycle
                    # Pruning: if len(path) + (n - len(visited)) <= len(best_cycle): return
                    # This is a valid upper bound on the max possible cycle length from this state.
                    if len(path) + (n - len(visited)) <= len(best_cycle):
                        return

                    for nb in sorted(adj[node], key=lambda x: -len(adj[x])):
                        if nb == start:
                            if len(path) >= 3:
                                if len(path) > len(best_cycle):
                                    best_cycle = path[:]
                            continue
                        
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            dfs(nb, path, visited)
                            path.pop()
                            visited.discard(nb)
                            # Early exit if we found a cycle that uses all nodes?
                            # If len(best_cycle) == n: return
                            if len(best_cycle) == n:
                                return

                dfs(start, [start], {start})
                if len(best_cycle) == n:
                    break # Can't do better than a Hamiltonian cycle
                    
            return best_cycle

        # Main loop
        while edge_set:
            cycle = find_longest_cycle(adj, n)
            if not cycle:
                break
            pieces.append(cycle)
            remove_cycle(cycle)
            
        # Remaining edges
        for u, v in edge_set:
            pieces.append([u, v])
            
        return pieces
