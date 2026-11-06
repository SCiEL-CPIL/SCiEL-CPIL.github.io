import sys
sys.setrecursionlimit(10000)

def construct():
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_biconnected_components(n, edges):
        # Returns list of sets of edges
        adj = {i: [] for i in range(n)}
        edge_id = {}
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            edge_id[(u, v)] = i
            edge_id[(v, u)] = i

        disc = [-1] * n
        low = [-1] * n
        time = 0
        stack = []
        components = []
        
        def dfs(u):
            nonlocal time
            disc[u] = time
            low[u] = time
            time += 1
            children = 0
            for v, eid in adj[u]:
                if disc[v] == -1:
                    stack.append(eid)
                    children += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] >= disc[u]:
                        comp = []
                        while True:
                            e = stack.pop()
                            comp.append(e)
                            if e == eid:
                                break
                        components.append(comp)
                elif v != u and disc[v] < disc[u]:
                    stack.append(eid)
                    low[u] = min(low[u], disc[v])

        for u in range(n):
            if disc[u] == -1:
                dfs(u)
                if stack:
                    components.append(stack.copy())
                    stack.clear()

        # Convert edge indices to edge tuples
        result = []
        for comp in components:
            result.append({edges[e] for e in comp})
        return result

    def find_2_factor(n, edges):
        # Returns a list of edges forming a 2-factor (spanning 2-regular subgraph) or None
        # Backtracking search
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        deg = [0] * n
        selected = []
        
        # Sort vertices by degree to prune
        order = sorted(range(n), key=lambda x: len(adj[x]))
        
        def backtrack(idx):
            if idx == len(order):
                return all(d == 2 for d in deg)
            
            u = order[idx]
            if deg[u] == 2:
                return backtrack(idx + 1)
            
            # Try to add edges incident to u
            # We need to select (2 - deg[u]) edges from adj[u]
            # But we must ensure the other ends don't exceed degree 2
            
            candidates = [v for v in adj[u] if deg[v] < 2]
            
            # Heuristic: try candidates that are also "tight"
            # Just iterate
            
            # We need to pick k edges.
            # This is a combinatorial selection.
            # For small degree, it's fine.
            
            # Optimization: If deg[u] == 0, we must pick 2.
            # If deg[u] == 1, we must pick 1.
            
            needed = 2 - deg[u]
            
            # Generate combinations of candidates
            from itertools import combinations
            for combo in combinations(candidates, needed):
                # Check if adding these edges is valid
                valid = True
                for v in combo:
                    if deg[v] + 1 > 2:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Add edges
                for v in combo:
                    selected.append((u, v))
                    deg[u] += 1
                    deg[v] += 1
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for v in combo:
                    selected.pop()
                    deg[u] -= 1
                    deg[v] -= 1
            
            return False

        if backtrack(0):
            return selected
        return None

    def extract_cycles(n, edges):
        # Given a set of edges, extract all cycles.
        # Returns list of cycles (lists of vertices)
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        visited_edges = set()
        cycles = []
        
        for u in range(n):
            for v in adj[u]:
                if (min(u, v), max(u, v)) in visited_edges:
                    continue
                
                # DFS to find a cycle starting with edge (u, v)
                # We want simple cycles.
                # This is hard to do efficiently for all cycles.
                # But in a 2-factor, the components are cycles.
                # So if the input is a 2-factor, we can just find connected components.
                
                pass
                
        # Better: If the graph is a 2-factor, each component is a cycle.
        # Let's assume the input to this function is a 2-factor.
        # Find connected components.
        visited_nodes = set()
        for u in range(n):
            if u in visited_nodes or not adj[u]:
                continue
            # BFS/DFS to find component
            comp = []
            stack = [u]
            visited_nodes.add(u)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if nei not in visited_nodes:
                        visited_nodes.add(nei)
                        stack.append(nei)
            
            # The component is a cycle
            # Reconstruct order
            # In a 2-regular connected graph, there is a unique cycle order.
            # Start at comp[0], go to a neighbor, etc.
            cycle = [comp[0]]
            current = comp[0]
            prev = -1
            while True:
                neighbors = [x for x in adj[current] if x != prev]
                if not neighbors:
                    break
                next_node = neighbors[0]
                if next_node == comp[0]:
                    break
                cycle.append(next_node)
                prev = current
                current = next_node
            
            cycles.append(cycle)
        return cycles

    def decompose_component(n, edges):
        # Recursively decompose a component
        if not edges:
            return []
        
        # Try to find a 2-factor
        factor_edges = find_2_factor(n, edges)
        
        if factor_edges:
            # Extract cycles from the 2-factor
            cycles = extract_cycles(n, factor_edges)
            pieces = [c for c in cycles if len(c) >= 3]
            
            # Remove factor edges from the set
            remaining = set()
            factor_set = set()
            for u, v in factor_edges:
                factor_set.add((min(u, v), max(u, v)))
            
            for u, v in edges:
                e = (min(u, v), max(u, v))
                if e not in factor_set:
                    remaining.add(e)
            
            # Recurse on remaining
            sub_pieces = decompose_component(n, list(remaining))
            return pieces + sub_pieces
        else:
            # No 2-factor found. 
            # Fall back to finding a single cycle (greedy) or single edges.
            # For robustness, let's just find one cycle using DFS.
            adj = {i: set() for i in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            visited = [False] * n
            parent = [-1] * n
            path = []
            cycle_found = None
            
            def dfs(u):
                nonlocal cycle_found
                if cycle_found:
                    return
                visited[u] = True
                path.append(u)
                for v in adj[u]:
                    if v == path[0] and len(path) >= 3:
                        cycle_found = path[:]
                        return
                    if not visited[v]:
                        dfs(v)
                        if cycle_found:
                            return
                path.pop()
            
            for start in range(n):
                if adj[start]:
                    dfs(start)
                    if cycle_found:
                        break
            
            if cycle_found:
                # Remove cycle edges
                cycle_set = set()
                for i in range(len(cycle_found)):
                    u = cycle_found[i]
                    v = cycle_found[(i + 1) % len(cycle_found)]
                    cycle_set.add((min(u, v), max(u, v)))
                
                remaining = set()
                for u, v in edges:
                    e = (min(u, v), max(u, v))
                    if e not in cycle_set:
                        remaining.add(e)
                
                return [cycle_found] + decompose_component(n, list(remaining))
            else:
                # No cycles, return single edges
                return [[u, v] for u, v in edges]

    def decompose(edges):
        if not edges:
            return []
        n = max(max(u, v) for u, v in edges) + 1
        components = get_biconnected_components(n, edges)
        
        all_pieces = []
        for comp_edges in components:
            if not comp_edges:
                continue
            # The component might not span all 0..n-1, but we can just pass the edges
            # find_2_factor needs correct n.
            # We can pass the local n or just the global n.
            pieces = decompose_component(n, list(comp_edges))
            all_pieces.extend(pieces)
            
        return all_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
