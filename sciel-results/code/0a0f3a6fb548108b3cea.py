import numpy as np

N = 16

def construct():
    # Build edge list in lexicographic order
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            edges.append((u, v))
    
    num_edges = len(edges)
    
    # Map (u, v) to edge index
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Precompute for each edge i, the list of (e1, e2, e3) tuples
    # where (e1, e2, e3) are the other 3 edges in a 4-cycle with edge i.
    # We only need cycles where edge i is the "last" edge (highest index)
    # for the greedy guarantee, but we'll store all and filter by index.
    
    cycles_for_edge = [[] for _ in range(num_edges)]
    
    for i, (u, v) in enumerate(edges):
        others = [k for k in range(N) if k != u and k != v]
        for xi in range(len(others)):
            x = others[xi]
            for yi in range(xi + 1, len(others)):
                y = others[yi]
                # Two 4-cycles containing edge (u,v) with vertices {u,v,x,y}:
                # Cycle A: u-v-x-y-u => edges: (u,v), (v,x), (x,y), (y,u)
                # Cycle B: u-v-y-x-u => edges: (u,v), (v,y), (y,x), (x,u)
                e1 = edge_idx[(v, x)]
                e2 = edge_idx[(x, y)]
                e3 = edge_idx[(y, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                e1 = edge_idx[(v, y)]
                e2 = edge_idx[(y, x)]
                e3 = edge_idx[(x, u)]
                cycles_for_edge[i].append((e1, e2, e3))
    
    # Greedy coloring
    colors = [0] * num_edges
    next_color = 0
    
    for i in range(num_edges):
        # Find all cycles where the other 3 edges are already colored (index < i)
        # and have 3 distinct colors. For those, our color must be in that set.
        forced_set = None
        
        for (e1, e2, e3) in cycles_for_edge[i]:
            if e1 < i and e2 < i and e3 < i:
                c1, c2, c3 = colors[e1], colors[e2], colors[e3]
                if len({c1, c2, c3}) == 3:
                    s = {c1, c2, c3}
                    if forced_set is None:
                        forced_set = s
                    else:
                        forced_set &= s
        
        if forced_set is not None and len(forced_set) > 0:
            # Must pick from forced_set; pick smallest for determinism
            colors[i] = min(forced_set)
        else:
            # No constraint from fully-colored cycles; use a new color to maximize diversity
            colors[i] = next_color
            next_color += 1
    
    return colors
