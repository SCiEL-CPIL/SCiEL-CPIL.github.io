import itertools
import random
import time

# EVOLVE-BLOCK-START
def construct():
    n = 20
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = []
    best_count = 0
    start_time = time.time()
    max_time = 100  # Safety margin
    
    # Helper to check if a specific edge (u,v,c) can be added to current graph
    # graph is a list of [u, v, c]
    # We need to check all 4-cycles containing (u,v).
    # A 4-cycle is u-v-w-x-u.
    # Edges: (u,v), (v,w), (w,x), (x,u).
    # We need colors c_uv, c_vw, c_wx, c_xu to be distinct.
    
    def can_add_edge(graph, u, v, c):
        # Build adjacency map for quick lookup
        # adj[x] = {y: color}
        # But graph is a list, so we rebuild or maintain.
        # For speed in this small graph, we can just check existing edges.
        
        # Find all w such that (v,w) is in graph
        # Find all x such that (x,u) is in graph and (w,x) is in graph
        # This is O(E^3) worst case if done naively per edge.
        # Better: Pre-compute neighbors.
        
        # Let's use a dictionary for edges: edge -> color
        # And neighbors: node -> set of neighbors
        
        # Since this is called inside a loop, we should pass the state
        # But for simplicity in the candidate, we'll do a slightly slower check
        # given n=20, E~100, it's fine.
        
        pass # Placeholder, will implement in main loop

    # Actually, let's use a faster check structure.
    # Maintain:
    # edge_color[(u,v)] = c
    # adj[u] = {v: c}
    
    def is_rainbow(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4

    def check_conflict(adj, edge_color, u, v, c_new):
        # Check all 4-cycles containing edge (u,v)
        # Cycle: u-v-w-x-u
        # 1. (v,w) exists
        # 2. (w,x) exists
        # 3. (x,u) exists
        
        # Iterate w in adj[v]
        #   Iterate x in adj[u]
        #     If x in adj[w]:
        #        Check colors
        
        for w in adj[v]:
            cw = edge_color[(v, w)] if (v, w) in edge_color else edge_color[(w, v)]
            # Check if w == u (triangle, ignore) or w == v
            if w == u: continue
            for x in adj[u]:
                cx = edge_color[(u, x)] if (u, x) in edge_color else edge_color[(x, u)]
                if x == v: continue
                if x == w: continue # Triangle
                
                # Check if (w,x) exists
                if x in adj[w]:
                    cwx = edge_color[(w, x)] if (w, x) in edge_color else edge_color[(x, w)]
                    # Colors: c_new (u-v), cw (v-w), cwx (w-x), cx (x-u)
                    if not is_rainbow(c_new, cw, cwx, cx):
                        return True # Conflict found
        
        return False # No conflict

    def run_greedy():
        adj = {i: set() for i in range(n)}
        edge_color = {}
        edge_list = []
        
        # Shuffle edges
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            # Try all colors
            found = False
            for c in range(20):
                if not check_conflict(adj, edge_color, u, v, c):
                    # Add edge
                    edge_color[(u, v)] = c
                    edge_color[(v, u)] = c
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_list.append([u, v, c])
                    found = True
                    break
            # If no color works, skip edge
            
        return edge_list

    best = []
    while time.time() - start_time < max_time:
        candidate = run_greedy()
        if len(candidate) > len(best):
            best = candidate
    
    return best
# EVOLVE-BLOCK-END
