import random
import time

def construct():
    n = 40
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    # Function to check if adding (u, v) creates a C6
    def creates_c6(u, v, adj):
        # Fast check using split paths
        if len(adj[u]) < 2 or len(adj[v]) < 2:
            return False
            
        # Paths of length 2 from u: (n1, n2)
        paths_u = []
        for x in adj[u]:
            for y in adj[x]:
                if y != u and y != x:
                    paths_u.append((x, y))
                    
        # Paths of length 2 from v: (n4, n3)
        paths_v = []
        for z in adj[v]:
            for w in adj[z]:
                if w != v and w != z:
                    paths_v.append((z, w))
                    
        for n1, n2 in paths_u:
            for n4, n3 in paths_v:
                if n2 == n3:
                    continue
                # Check if edge (n2, n3) exists
                # O(D) lookup
                if n3 in adj[n2]:
                    # Check distinctness of all 6 vertices: u, n1, n2, n3, n4, v
                    # n1 != u, n2 != u, n2 != n1
                    # n4 != v, n3 != v, n3 != n4
                    # n2 != n3
                    # Cross checks:
                    if n1 != n3 and n1 != n4 and n2 != n4:
                        if n3 != u and n4 != u and n1 != v and n2 != v:
                            return True
                            
        return False

    # Greedy construction
    def greedy_random():
        nonlocal best_edges, best_count
        
        random.shuffle(all_edges)
        adj = [[] for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if not creates_c6(u, v, adj):
                adj[u].append(v)
                adj[v].append(u)
                current_edges.append((u, v))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    # Run multiple restarts
    while time.time() < deadline:
        greedy_random()
        
    return [[u, v] for u, v in best_edges]
