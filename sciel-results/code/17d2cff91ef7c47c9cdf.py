import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    
    # Start with known good: two K3,3 components
    # K3,3 on vertices 0..5: part A = {0,1,2}, part B = {3,4,5}
    # K3,3 on vertices 6..11: part A = {6,7,8}, part B = {9,10,11}
    best_edges = []
    for i in range(3):
        for j in range(3):
            best_edges.append([i, 3 + j])
            best_edges.append([6 + i, 9 + j])
    best_score = 2  # K3,3 has cycle lengths {4, 6}
    
    # Try to find a graph with 1 distinct cycle length
    # Strategy: search over 3-regular bipartite graphs on 12 vertices
    # Parts: A={0..5}, B={6..11}
    # A 3-regular bipartite graph can be represented by a 6x6 0-1 matrix with row/col sums 3
    
    import random
    random.seed(42)
    
    def get_cycle_lengths(edges):
        from collections import defaultdict
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        def find_cycles_from(start):
            found = []
            def dfs(node, path, visited):
                for neighbor in sorted(adj[node]):
                    if neighbor == start and len(path) >= 4:  # Girth >= 4, so no triangles
                        found.append(len(path))
                    elif neighbor not in visited and neighbor > start:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            dfs(start, [start], {start})
            return found
        
        for v in range(n):
            for c in find_cycles_from(v):
                cycle_lengths.add(c)
        
        return cycle_lengths
    
    def is_valid(edges):
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d < 3:
                return False
        
        from collections import defaultdict
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # Check has cycle (using Union-Find)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return True
            parent[px] = py
            return False
        
        has_cycle = False
        for u, v in edges:
            if union(u, v):
                has_cycle = True
                break
        
        return has_cycle
    
    # Generate random 3-regular bipartite graphs properly
    # Use configuration model: create stubs and pair them randomly
    for attempt in range(50000):
        if time.time() > deadline - 5:
            break
        
        # Configuration model for 3-regular bipartite graph
        # A vertices 0-5 each have 3 stubs, B vertices 6-11 each have 3 stubs
        # Randomly pair A-stubs with B-stubs
        
        a_stubs = [(i, s) for i in range(6) for s in range(3)]
        b_stubs = [(6+i, s) for i in range(6) for s in range(3)]
        
        random.shuffle(b_stubs)
        
        edges = set()
        valid = True
        for (a, _), (b, _) in zip(a_stubs, b_stubs):
            key = (min(a, b), max(a, b))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if not valid:
            continue
        
        edge_list = [list(e) for e in edges]
        
        if not is_valid(edge_list):
            continue
        
        cl = get_cycle_lengths(edge_list)
        if not cl:
            continue
        
        score = len(cl)
        if score < best_score:
            best_score = score
            best_edges = edge_list
            if score == 1:
                break
    
    # Final: ensure types are correct
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
    
    return result
