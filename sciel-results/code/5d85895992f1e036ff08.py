import random
import time
import sys

def construct():
    n = 20
    colors = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # State: graph as set of (u, v, c) with u < v
    # We want to maximize edges.
    # Strategy: 
    # 1. Start with a complete graph K20? No, that has many 4-cycles.
    #    Actually, let's start with a sparse graph and add edges greedily.
    #    Or start with a dense graph and remove violating edges.
    #    Removing is safer to ensure validity.
    
    # Let's try starting with a random dense graph and fixing violations.
    # A violation is a 4-cycle with duplicate colors.
    
    # Helper: Check if adding edge (u, v, c) creates a non-rainbow 4-cycle
    # A 4-cycle involves u-v-w-x-u.
    # If we add (u, v, c), we need to check all paths u-w-x-v of length 2?
    # No, a 4-cycle is u-v-w-x-u. The new edge is (u,v).
    # The other 3 edges are (v,w), (w,x), (x,u).
    # So for any pair (w, x) such that (v,w) in E, (w,x) in E, (x,u) in E,
    # we form a cycle u-v-w-x-u.
    # The colors must be distinct: c, c(v,w), c(w,x), c(x,u) all distinct.
    
    # This check is O(N^2) per edge addition if done naively.
    # N=20, so N^2=400. Total edges ~200. 200*400 = 80,000 ops. Very fast.
    
    # Let's implement a greedy adder.
    # Order of edges: Random or structured?
    # Structured might be better for density.
    # Let's try all pairs (u, v) with u < v.
    # For each pair, try to assign a color.
    # If we assign a color, we must ensure no 4-cycle is broken.
    
    # Better approach: 
    # 1. Initialize empty graph.
    # 2. Iterate over all possible edges (u, v) in a random order.
    # 3. For each (u, v), try to find a color c in 0..19 such that adding (u, v, c)
    #    does not create a non-rainbow 4-cycle.
    # 4. If such a color exists, add the edge with that color.
    # 5. If no color works, skip the edge.
    
    # To improve, we can use Local Search:
    # After greedy pass, try to re-color existing edges to allow adding more edges.
    
    def get_neighbors(v, graph):
        # Returns dict: neighbor -> color
        return {u: c for (a, b, c) in graph if a == v or b == v for u in [a, b] if u != v}
        # Wait, graph is set of (u, v, c). 
        # Let's use a dict: (u, v) -> c for u < v.
        
    # Let's use a dict for the graph: key (u, v) with u < v, value color.
    graph = {}
    
    def is_valid_addition(u, v, c, graph):
        # Check if adding (u, v, c) creates a non-rainbow 4-cycle.
        # A 4-cycle containing (u, v) must be u-v-w-x-u.
        # w is a neighbor of v, x is a neighbor of u.
        # And (w, x) must be an edge.
        
        # Get neighbors of v
        nbrs_v = {}
        for (a, b), col in graph.items():
            if a == v: nbrs_v[b] = col
            elif b == v: nbrs_v[a] = col
            
        # Get neighbors of u
        nbrs_u = {}
        for (a, b), col in graph.items():
            if a == u: nbrs_u[b] = col
            elif b == u: nbrs_u[a] = col
            
        # For each w in nbrs_v, and each x in nbrs_u:
        # If (w, x) is in graph, then u-v-w-x-u is a 4-cycle.
        # Colors: c (for u-v), col_vw (for v-w), col_wx (for w-x), col_xu (for x-u).
        # They must be all distinct.
        
        for w, col_vw in nbrs_v.items():
            if w == u: continue # Not a simple 4-cycle if w=u? No, w is neighbor of v, u is endpoint.
            for x, col_xu in nbrs_u.items():
                if x == v: continue
                if w == x: continue # Not a simple 4-cycle
                # Check if (w, x) is an edge
                key_wx = (min(w, x), max(w, x))
                if key_wx in graph:
                    col_wx = graph[key_wx]
                    # Check distinctness
                    cols = [c, col_vw, col_wx, col_xu]
                    if len(set(cols)) < 4:
                        return False
        return True

    def find_valid_color(u, v, graph):
        for c in range(colors):
            if is_valid_addition(u, v, c, graph):
                return c
        return -1

    # Greedy Phase
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        c = find_valid_color(u, v, graph)
        if c != -1:
            graph[(u, v)] = c
    
    # Local Search Phase: Try to add skipped edges by re-coloring
    # Or try to re-color existing edges to free up color slots for new edges.
    
    # Simple Local Search: 
    # Iterate over edges not in graph. Try to add them.
    # If can't add, try to re-color one of the edges in a conflicting cycle to allow addition.
    
    # This is complex. Let's try a simpler Local Search:
    # Randomly pick an edge in the graph, remove it, try to re-add it with a different color?
    # No, that doesn't increase edge count.
    
    # Better Local Search:
    # Pick a missing edge (u, v).
    # Find a conflicting 4-cycle.
    # Try to re-color an edge in that cycle to resolve the conflict.
    
    # Given time constraints, let's stick to the greedy result if it's good.
    # But we can do a quick random restart if time permits.
    
    best_edges = len(graph)
    best_graph = dict(graph)
    
    # Try a few random restarts of the greedy phase
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        graph = {}
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            c = find_valid_color(u, v, graph)
            if c != -1:
                graph[(u, v)] = c
                
        if len(graph) > best_edges:
            best_edges = len(graph)
            best_graph = dict(graph)
            
    # Convert best_graph to list format
    result = []
    for (u, v), c in best_graph.items():
        result.append([u, v, c])
        
    return result
