import numpy as np

N = 16

def construct():
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            edges.append((u, v))
            
    num_edges = len(edges)
    
    # Map (u, v) to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    # Precompute cycles for each edge
    # cycles_for_edge[i] contains tuples (e1, e2, e3)
    # where (e1, e2, e3) are the other 3 edges in a 4-cycle containing edge i.
    # We only care about cycles where i is the "last" edge in terms of processing order?
    # No, we precompute all cycles containing i.
    # During coloring, we check if e1, e2, e3 are already colored.
    
    cycles_for_edge = [[] for _ in range(num_edges)]
    
    for i, (u, v) in enumerate(edges):
        others = [k for k in range(N) if k != u and k != v]
        for idx_x in range(len(others)):
            x = others[idx_x]
            for idx_y in range(idx_x + 1, len(others)):
                y = others[idx_y]
                # Cycle 1: u-v-x-y-u
                # Edges: (u,v), (v,x), (x,y), (y,u)
                e1 = edge_to_idx[(v, x)]
                e2 = edge_to_idx[(x, y)]
                e3 = edge_to_idx[(y, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                # Cycle 2: u-v-y-x-u
                # Edges: (u,v), (v,y), (y,x), (x,u)
                e1 = edge_to_idx[(v, y)]
                e2 = edge_to_idx[(y, x)]
                e3 = edge_to_idx[(x, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
    colors = [-1] * num_edges
    next_color = 0
    color_counts = {}
    
    for i in range(num_edges):
        allowed_set = None
        
        for (e1, e2, e3) in cycles_for_edge[i]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            
            # Check if all 3 are colored
            if c1 != -1 and c2 != -1 and c3 != -1:
                s = {c1, c2, c3}
                if len(s) == 3:
                    if allowed_set is None:
                        allowed_set = s
                    else:
                        allowed_set = allowed_set.intersection(s)
                        
        if allowed_set is not None:
            # Must pick a color from allowed_set
            # Heuristic: Pick the color with the lowest count so far to balance usage?
            # Or just the smallest index?
            # Let's try smallest index first as it's deterministic and simple.
            # Actually, picking the least used color might be better for maximizing diversity in the long run?
            # If we pick a color that is already very common, it might force more edges to reuse it?
            # If we pick a rare color, it might stay rare?
            # Let's stick to min(allowed_set) for now, it's a standard greedy choice.
            c = min(allowed_set)
            colors[i] = c
            color_counts[c] = color_counts.get(c, 0) + 1
        else:
            # Can use a new color
            colors[i] = next_color
            color_counts[next_color] = 1
            next_color += 1
            
    return colors
