import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 20
    num_colors = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    def try_construction(seed_val):
        random.seed(seed_val)
        edges = all_edges[:]
        random.shuffle(edges)
        
        # graph[u][v] = color
        graph = {}
        adj = [set() for _ in range(n)]
        
        count = 0
        
        for u, v in edges:
            # Check if we can add (u, v)
            # Find all 4-cycles u-x-y-v that would be completed
            # x in adj[u], y in adj[v], (x,y) in graph
            
            forbidden = set()
            
            # Iterate x in neighbors of u
            for x in adj[u]:
                # Iterate y in neighbors of v
                for y in adj[v]:
                    if x == y:
                        continue
                    # Check if (x,y) is an edge
                    if x in graph and y in graph[x]:
                        # We have a path u-x-y-v
                        # Colors: c(u,x), c(x,y), c(y,v)
                        c1 = graph[u][x]
                        c2 = graph[x][y]
                        c3 = graph[y][v]
                        forbidden.add(c1)
                        forbidden.add(c2)
                        forbidden.add(c3)
                        
            if len(forbidden) < num_colors:
                # Find a valid color
                valid_color = None
                for c in range(num_colors):
                    if c not in forbidden:
                        valid_color = c
                        break
                
                if valid_color is not None:
                    # Add edge
                    if u not in graph:
                        graph[u] = {}
                    if v not in graph:
                        graph[v] = {}
                    graph[u][v] = valid_color
                    graph[v][u] = valid_color
                    adj[u].add(v)
                    adj[v].add(u)
                    count += 1
        
        return count, graph

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        seed_val = restarts
        count, graph = try_construction(seed_val)
        
        if count > best_count:
            best_count = count
            # Convert graph to list of edges
            best_edges = []
            for u in range(n):
                if u in graph:
                    for v in graph[u]:
                        if u < v:
                            best_edges.append([u, v, graph[u][v]])
        
        # Optional: If we reach a very high count, we could break, 
        # but let's use the time to find potentially even better ones.
        
    return best_edges
