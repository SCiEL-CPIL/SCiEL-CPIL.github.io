import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n = 16
    # List of all edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)
    
    # Adjacency matrix for fast lookup
    adj = [[False]*n for _ in range(n)]
    
    def is_q3(subset):
        # subset is a list of 8 vertices
        # Check if induced subgraph contains Q3
        # This is a subgraph isomorphism problem.
        # For 8 vertices, we can brute force permutations? 8! = 40320.
        # But Q3 has symmetries.
        # We can check if the induced subgraph has 12 edges and is isomorphic to Q3.
        # Actually, just check if it CONTAINS Q3.
        # A graph on 8 vertices contains Q3 if it has a subgraph isomorphic to Q3.
        # Since Q3 has 12 edges, if induced subgraph has < 12 edges, return False.
        
        # Count edges in induced subgraph
        e_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    e_count += 1
        if e_count < 12:
            return False
            
        # Check isomorphism to Q3
        # Q3 is 3-regular bipartite.
        # If induced subgraph is 3-regular and bipartite, it is Q3?
        # There is only one 3-regular bipartite graph on 8 vertices: Q3.
        # So check if degrees are all 3 and it is bipartite.
        
        # Check degrees
        for i in range(8):
            deg = 0
            for j in range(8):
                if i != j and adj[subset[i]][subset[j]]:
                    deg += 1
            if deg != 3:
                return False
                
        # Check bipartite
        # BFS/DFS coloring
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if u != v and adj[subset[u]][subset[v]]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    def find_q3():
        # Find any Q3 in current graph
        # Random sampling of 8 vertices
        for _ in range(1000):
            subset = random.sample(range(n), 8)
            if is_q3(subset):
                return subset
        return None

    # Phase 1: Remove edges until Q3-free
    # Start with complete graph
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    removed_edges = set()
    
    while time.time() < deadline:
        q3_subset = find_q3()
        if q3_subset is None:
            break
        # Remove a random edge from this Q3
        # Get all edges in the induced subgraph
        sub_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if adj[q3_subset[i]][q3_subset[j]]:
                    sub_edges.append((q3_subset[i], q3_subset[j]))
        if sub_edges:
            e_to_remove = random.choice(sub_edges)
            adj[e_to_remove[0]][e_to_remove[1]] = False
            adj[e_to_remove[1]][e_to_remove[0]] = False
            removed_edges.add(e_to_remove)
            
    # Phase 2: Try to add edges back
    # Sort missing edges
    missing_edges = [e for e in edges if e not in removed_edges and not adj[e[0]][e[1]]]
    # Actually, just iterate all edges not in graph
    for u, v in edges:
        if time.time() > deadline:
            break
        if adj[u][v]:
            continue
        # Try adding (u,v)
        adj[u][v] = True
        adj[v][u] = True
        
        # Check if this creates a Q3
        # Only need to check Q3s containing (u,v)
        # This is hard to do efficiently without precomputation.
        # So we do a global check?
        # If we do global check, it's slow.
        # Let's just do a few random checks.
        is_valid = True
        for _ in range(50):
            subset = random.sample(range(n), 8)
            if u in subset and v in subset:
                if is_q3(subset):
                    is_valid = False
                    break
        if not is_valid:
            adj[u][v] = False
            adj[v][u] = False
        else:
            removed_edges.discard((u,v))
            removed_edges.discard((v,u))

    # Count edges
    count = 0
    for u, v in edges:
        if adj[u][v]:
            count += 1
            
    return [[u, v] for u, v in edges if adj[u][v]]

construct()
