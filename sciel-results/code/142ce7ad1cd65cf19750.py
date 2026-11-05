"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)
    
    # Helper to check if adding edge (u,v) with color c creates a non-rainbow 4-cycle
    # Given current graph state (adj, edge_colors), check all 4-cycles containing (u,v)
    def is_valid_addition(u, v, c, adj, edge_colors):
        # A 4-cycle containing edge (u,v) has the form u-v-x-y-u
        # where x is a neighbor of v (x != u), y is a neighbor of u (y != v),
        # x != y, and x-y is an edge.
        # The four edges are: (u,v), (v,x), (x,y), (y,u)
        # We need all four colors to be distinct.
        
        for x in adj[v]:
            if x == u:
                continue
            # Edge (v,x) must exist (it does since x in adj[v])
            c_vx = edge_colors[(min(v,x), max(v,x))]
            if c_vx == c:
                return False  # (u,v) and (v,x) would have same color
            for y in adj[u]:
                if y == v or y == x:
                    continue
                # Check if x-y is an edge
                if y not in adj[x]:
                    continue
                c_xy = edge_colors[(min(x,y), max(x,y))]
                c_yu = edge_colors[(min(y,u), max(y,u))]
                
                # Check if any of the four edges have the same color
                if c_xy == c or c_yu == c or c_xy == c_vx or c_yu == c_vx:
                    return False
        
        return True
    
    # Greedy with multiple restarts and different edge orderings
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 13)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Track current graph
        adj = {i: set() for i in range(n)}
        edge_colors = {}
        current_edges = []
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            
            # Try to find a valid color for this edge
            valid_color = None
            # Try colors in a randomized order for diversity
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                if is_valid_addition(u, v, c, adj, edge_colors):
                    valid_color = c
                    break
            
            if valid_color is not None:
                adj[u].add(v)
                adj[v].add(u)
                edge_colors[(u, v)] = valid_color
                current_edges.append([u, v, valid_color])
            
            if len(current_edges) > best_count:
                best_edges = current_edges[:]
                best_count = len(current_edges)
        
        if len(current_edges) > best_count:
            best_edges = current_edges[:]
            best_count = len(current_edges)
    
    return best_edges

# EVOLVE-BLOCK-END
