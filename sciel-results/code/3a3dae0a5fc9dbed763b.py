import time
import itertools

def construct():
    start_time = time.time()
    deadline = 58.0
    
    n = 12
    edges = []
    
    # Construct two disjoint K_{3,3} components
    # Component 1: Vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    # Component 2: Vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    
    # Generate edges for K_{3,3} on 0-5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u, v in itertools.product(part_a1, part_b1):
        edges.append([u, v])
        
    # Generate edges for K_{3,3} on 6-11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u, v in itertools.product(part_a2, part_b2):
        edges.append([u, v])
        
    # Validation
    if time.time() - start_time > deadline:
        return []
        
    edge_set = set()
    for e in edges:
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int):
            return []
        if isinstance(u, bool) or isinstance(v, bool):
            return []
        if u < 0 or u >= n or v < 0 or v >= n:
            return []
        if u == v:
            return []
        if (u, v) in edge_set or (v, u) in edge_set:
            return []
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Check degree
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    if min(degrees) < 3:
        return []
        
    # Check Triangle Free (Bipartite)
    # Since we constructed bipartite components, it is triangle free.
    # We verify by checking if any edge connects nodes in the same "part" 
    # or by BFS coloring.
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    color = [-1] * n
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            stack = [i]
            while stack:
                node = stack.pop()
                for nei in adj[node]:
                    if color[nei] == -1:
                        color[nei] = 1 - color[node]
                        stack.append(nei)
                    elif color[nei] == color[node]:
                        return []
                        
    return edges
