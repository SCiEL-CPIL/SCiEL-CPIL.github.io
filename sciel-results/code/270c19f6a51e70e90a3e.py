import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Construct Grötzsch graph on vertices 0..10
    # Vertices 0-4 form a cycle C5
    # Vertices 5-9 are connected to u_{i-1} and u_{i+1}
    # Vertex 10 is connected to all v_i (5-9)
    
    adj = [set() for _ in range(16)]
    
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        u = i
        v = (i + 1) % 5
        adj[u].add(v)
        adj[v].add(u)
        
    # v_i (5+i) connected to u_{i-1} and u_{i+1}
    for i in range(5):
        v = 5 + i
        u_prev = (i - 1) % 5
        u_next = (i + 1) % 5
        adj[v].add(u_prev)
        adj[u_prev].add(v)
        adj[v].add(u_next)
        adj[u_next].add(v)
        
    # w (10) connected to all v_i
    for i in range(5):
        v = 5 + i
        adj[10].add(v)
        adj[v].add(10)
        
    # Vertices 11-15 are initially isolated
    
    # Generate all possible edges
    all_edges = list(combinations(range(16), 2))
    
    best_edge_count = -1
    best_adj = None
    
    # Random restarts to find a dense maximal triangle-free supergraph
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle edges
        random.shuffle(all_edges)
        
        # Copy adjacency list
        curr_adj = [set(s) for s in adj]
        
        # Greedily add edges
        for u, v in all_edges:
            # Check if edge already exists
            if v in curr_adj[u]:
                continue
            # Check if adding edge creates a triangle
            # i.e., if u and v share a common neighbor
            if curr_adj[u] & curr_adj[v]:
                continue
            # Add edge
            curr_adj[u].add(v)
            curr_adj[v].add(u)
            
        # Count edges
        edge_count = sum(len(s) for s in curr_adj) // 2
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = curr_adj
    
    # Convert best_adj to list of edges
    edges = []
    for u in range(16):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges
