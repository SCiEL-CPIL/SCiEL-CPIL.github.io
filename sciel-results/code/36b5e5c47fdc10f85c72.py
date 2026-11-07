import time
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit
    
    # Model: Two disjoint copies of K_{3,3}
    # K_{3,3} is bipartite with parts of size 3.
    # Cycle lengths in K_{3,3} are 4 and 6.
    # Thus, the union has distinct cycle lengths {4, 6}.
    # Count = 2. This is the minimum possible for min-degree 3, girth >= 4.
    
    edges = []
    
    # Component 1: K_{3,3} on vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Verification and Optimization (Minimal)
    n = 12
    degree = [0] * n
    edge_set = set()
    
    for u, v in edges:
        # Type checks
        if not isinstance(u, int) or not isinstance(v, int):
            return []
        if isinstance(u, bool) or isinstance(v, bool):
            return []
            
        # Range checks
        if u < 0 or u >= n or v < 0 or v >= n:
            return []
            
        # Loop check
        if u == v:
            return []
            
        # Duplicate check
        if (u, v) in edge_set or (v, u) in edge_set:
            return []
            
        edge_set.add((u, v))
        edge_set.add((v, u))
        degree[u] += 1
        degree[v] += 1
        
    # Check min degree
    if any(d < 3 for d in degree):
        return []
        
    # Check triangle-free (bipartite check)
    # Since we constructed bipartite components, it is triangle-free.
    # We perform a quick BFS 2-coloring to be absolutely sure.
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    color = [-1] * n
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            queue = [i]
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return []  # Odd cycle found (triangle or other)
                        
    # Check for cycles
    # K3,3 has cycles. 
    # Verify graph is not a forest.
    # With n=12 and min degree 3, edges >= 18. Forest has <= 11 edges.
    # So cycles exist.
    
    return edges

# EVOLVE-BLOCK-START
# The above code is the entire program.
# EVOLVE-BLOCK-END
