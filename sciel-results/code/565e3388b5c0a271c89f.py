import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 10s safety margin
    
    best_edges = []
    best_count = 0
    
    # Helper to check if edge (u, v) can be added
    # adj[u] is a set of neighbors
    # c[u][v] is count of common neighbors
    
    def try_add_edges(adj, c, edges):
        # Generate random permutation of all possible edges
        possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(possible_edges)
        
        current_edges = []
        current_adj = [set() for _ in range(n)]
        current_c = [[0] * n for _ in range(n)]
        
        for u, v in possible_edges:
            if time.time() > deadline:
                break
            
            # Check if adding (u, v) violates K2,3
            # 1. c[u][v] must be <= 2
            if current_c[u][v] > 2:
                continue
                
            # 2. For all w in N(u), c[w][v] must be < 2
            #    Because adding (u,v) adds u to N(v), so u becomes a common neighbor of w and v.
            #    Wait, c[w][v] counts common neighbors of w and v.
            #    If we add edge (u,v), u becomes a neighbor of v.
            #    If u is already a neighbor of w, then u is a common neighbor of w and v.
            #    So c[w][v] increases by 1 for all w in N(u) \cap N(v)? No.
            #    Let's re-evaluate.
            #    Common neighbors of w and v are vertices x such that x~w and x~v.
            #    If we add edge (u,v), then u is now a neighbor of v.
            #    If u is also a neighbor of w, then u becomes a new common neighbor of w and v.
            #    So c[w][v] increases by 1 IF u is in N(w).
            #    Since u is in N(w) is equivalent to w in N(u) (symmetry).
            #    So for all w in N(u), c[w][v] increases by 1.
            #    We need c[w][v] + 1 <= 2 => c[w][v] <= 1.
            
            valid = True
            for w in current_adj[u]:
                if current_c[w][v] > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            for w in current_adj[v]:
                if current_c[u][w] > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add edge
            current_edges.append([u, v])
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_c[u][v] += 1
            current_c[v][u] += 1
            
            for w in current_adj[u]:
                current_c[w][v] += 1
                current_c[v][w] += 1
            
            for w in current_adj[v]:
                current_c[u][w] += 1
                current_c[w][u] += 1
        
        return current_edges

    # Run multiple iterations
    iterations = 0
    while time.time() < deadline:
        edges = try_add_edges(None, None, None)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        iterations += 1
        
        # If we've done a reasonable number of iterations and are getting stuck, break early?
        # Or just keep going until deadline.
        # Each iteration is O(E_max * N) roughly. 20 vertices, ~100 edges. Very fast.
        # We can do thousands of iterations.

    return best_edges
