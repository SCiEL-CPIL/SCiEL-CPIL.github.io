def construct():
    # 1. All odds
    odds = [i for i in range(1, 257) if i % 2 != 0]
    
    # 2. Candidate safe evens
    safe_evens = []
    for i in range(2, 257, 2):
        # Calculate v2 and odd part
        x = i
        v2 = 0
        while x % 2 == 0:
            x //= 2
            v2 += 1
        odd_part = x
        if odd_part <= (1 << v2):
            safe_evens.append(i)
            
    # 3. Build conflict graph for safe_evens
    n = len(safe_evens)
    conflict = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a = safe_evens[i]
            b = safe_evens[j]
            if (a * b) % (a + b) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # 4. Find MIS of this graph
    # Simple backtracking
    def get_mis(nodes, graph):
        # nodes: list of indices
        # graph: adjacency matrix
        if not nodes:
            return []
        
        # Pick a node with highest degree to branch
        # Or just pick the first one for simplicity, 23 nodes is small
        v = nodes[0]
        rest = nodes[1:]
        
        # Option 1: Exclude v
        mis_exclude = get_mis(rest, graph)
        
        # Option 2: Include v
        # Must exclude all neighbors of v
        neighbors = [u for u in rest if graph[v][u]]
        non_neighbors = [u for u in rest if not graph[v][u]]
        mis_include = [v] + get_mis(non_neighbors, graph)
        
        if len(mis_include) > len(mis_exclude):
            return mis_include
        else:
            return mis_exclude
            
    mis_indices = get_mis(list(range(n)), conflict)
    safe_mis = [safe_evens[i] for i in mis_indices]
    
    # 5. Combine
    result = odds + safe_mis
    return result
