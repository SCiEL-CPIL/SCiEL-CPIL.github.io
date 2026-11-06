import time
import random
import sys

def construct():
    n = 20
    colors_count = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute all 4-cycles as tuples of edge indices
    # Map (u,v) to index
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    cycles = []
    # Iterate over all sets of 4 vertices
    for i in range(n-3):
        for j in range(i+1, n-2):
            for k in range(j+1, n-1):
                for l in range(k+1, n):
                    u, v, w, x = i, j, k, l
                    # 3 distinct 4-cycles on {u,v,w,x}
                    # 1. u-v-w-x-u
                    try:
                        e1 = edge_index[(u, v)]
                        e2 = edge_index[(v, w)]
                        e3 = edge_index[(w, x)]
                        e4 = edge_index[(x, u)]
                        cycles.append((e1, e2, e3, e4))
                    except KeyError:
                        pass
                    
                    # 2. u-v-x-w-u
                    try:
                        e1 = edge_index[(u, v)]
                        e2 = edge_index[(v, x)]
                        e3 = edge_index[(x, w)]
                        e4 = edge_index[(w, u)]
                        cycles.append((e1, e2, e3, e4))
                    except KeyError:
                        pass
                    
                    # 3. u-w-v-x-u
                    try:
                        e1 = edge_index[(u, w)]
                        e2 = edge_index[(w, v)]
                        e3 = edge_index[(v, x)]
                        e4 = edge_index[(x, u)]
                        cycles.append((e1, e2, e3, e4))
                    except KeyError:
                        pass

    # For each edge, find which cycles it belongs to
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)

    # We want to select a set of edges and assign colors such that:
    # 1. No two edges share a vertex (independent set in line graph? No, we can have multiple edges per vertex, just not same color in 4-cycle)
    # Actually, the problem allows multiple edges incident to a vertex.
    # The constraint is only on 4-cycles.
    # However, if we select a set of edges, we must be able to color them.
    # This is equivalent to finding a subgraph H of K_n and a coloring c: E(H) -> {0..19} such that every 4-cycle in H is rainbow.
    
    # Strategy: Greedy construction of a valid graph.
    # We iterate through potential edges (u,v,c) in random order.
    # We maintain the set of selected edges (u,v) and their colors.
    # When considering adding (u,v,c), we check if it violates the rainbow condition with any existing 4-cycle.
    # A new edge (u,v,c) can only create new 4-cycles involving (u,v).
    # The 4-cycles involving (u,v) are of the form u-x-v-y-u or u-x-y-v-u?
    # No, a 4-cycle containing edge (u,v) must look like u-v-w-x-u.
    # The other three edges are (v,w), (w,x), (x,u).
    # So for a new edge (u,v,c), we need to check all pairs (w,x) such that (v,w), (w,x), (x,u) are in the current graph.
    # If such a path exists, the cycle is u-v-w-x-u.
    # The colors must be distinct: c, col(v,w), col(w,x), col(x,u) must be all different.
    
    # To make this fast, we can precompute for each pair (u,v) the list of "paths of length 3" connecting them via two intermediate nodes?
    # Or simpler: For each edge (u,v) we add, we check all w != u,v and x != u,v,w.
    # If (v,w), (w,x), (x,u) are present, check colors.
    # Complexity per edge addition: O(n^2) checks.
    # Total edges in K20 is 190.
    # 190 * 20^2 = 76000 operations. Very fast.
    
    # We will try multiple random orderings of the 190*20 = 3800 possible (edge, color) triples?
    # No, we just iterate through edges in random order, and for each edge, try all 20 colors.
    # If a color works, add it.
    # If no color works, skip the edge.
    
    # To maximize edges, we should prioritize edges that are "critical" or have high degree?
    # Or just random.
    # Let's try a few restarts with different random seeds/orders.
    
    best_edges = []
    best_count = 0
    
    for attempt in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        # Shuffle edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # State
        current_edges = {} # (u,v) -> color
        # Adjacency for fast lookup
        adj = {i: set() for i in range(n)}
        
        count = 0
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            
            # Try each color for this edge
            found_color = False
            # Shuffle colors too? Or try all?
            # Trying all 20 is cheap.
            for c in range(colors_count):
                # Check if adding (u,v,c) creates a non-rainbow 4-cycle
                valid = True
                # Any 4-cycle containing (u,v) is u-v-w-x-u
                # We need (v,w), (w,x), (x,u) to exist in current_edges
                # Iterate w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (v, w) not in current_edges and (w, v) not in current_edges:
                        continue
                    # Get color of (v,w)
                    if (v, w) in current_edges:
                        cw = current_edges[(v, w)]
                    else:
                        cw = current_edges[(w, v)]
                        
                    # Now need x such that (w,x) and (x,u) exist
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if (w, x) not in current_edges and (x, w) not in current_edges:
                            continue
                        if (x, u) not in current_edges and (u, x) not in current_edges:
                            continue
                        
                        # Get colors
                        if (w, x) in current_edges:
                            cxw = current_edges[(w, x)]
                        else:
                            cxw = current_edges[(x, w)]
                        
                        if (x, u) in current_edges:
                            cxu = current_edges[(x, u)]
                        else:
                            cxu = current_edges[(u, x)]
                        
                        # Cycle colors: c, cw, cxw, cxu
                        if len({c, cw, cxw, cxu}) < 4:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Add edge
                    current_edges[(u, v)] = c
                    adj[u].add(v)
                    adj[v].add(u)
                    count += 1
                    found_color = True
                    break
            
            if found_color:
                continue
            else:
                # Could not add edge (u,v) with any color
                # Skip
                pass
        
        if count > best_count:
            best_count = count
            best_edges = [(u, v, c) for (u, v), c in current_edges.items()]
            
            # If we reached max possible (190), we can stop early
            if best_count == 190:
                break

    # Convert to required format [[u, v, c], ...]
    # Ensure u < v for consistency? The problem says simple graph, [u,v,c].
    # Usually order doesn't matter for validity check if it checks subgraphs, but let's normalize.
    result = []
    for u, v, c in best_edges:
        if u > v:
            u, v = v, u
        result.append([u, v, c])
        
    # Fallback if nothing found (should not happen, star is always valid)
    if not result:
        result = [[0, v, 0] for v in range(1, n)]
        
    return result

# EVOLVE-BLOCK-END
