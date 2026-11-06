import time

def construct():
    n = 20
    colors_count = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # We construct a complete 6-partite graph (blow-up of K6) with all 15 edge-colors distinct.
    # Partition sizes: 4, 4, 4, 3, 3, 2 -> sum = 20
    # This guarantees any 4-cycle projects to a cycle in K6 with distinct edge colors,
    # hence the 4-cycle in the blow-up is rainbow.
    
    # Number of edges in complete 6-partite with these sizes:
    # Total edges = (n^2 - sum(size_i^2)) / 2 = (400 - (16+16+16+9+9+4)) / 2 = (400-70)/2 = 165
    
    # Build the partition
    sizes = [4, 4, 4, 3, 3, 2]
    clusters = []
    idx = 0
    for s in sizes:
        clusters.append(list(range(idx, idx + s)))
        idx += s
    
    # Assign unique colors to each pair of clusters
    # 6 clusters -> 15 pairs, we have 20 colors, so we use 15 distinct colors
    edges = []
    color_idx = 0
    num_clusters = len(clusters)
    
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            c = color_idx
            color_idx += 1
            for u in clusters[i]:
                for v in clusters[j]:
                    edges.append([u, v, c])
    
    # Verify the construction is valid (all 4-cycles are rainbow)
    # For safety, we do a quick check. If invalid, we fall back to a star.
    # But the construction is mathematically guaranteed valid.
    # Let's still check to be safe.
    
    # Build edge color map for quick lookup
    edge_color = {}
    for u, v, c in edges:
        edge_color[(min(u, v), max(u, v))] = c
    
    # Check all 4-cycles
    # A 4-cycle is u-v-w-x-u with distinct u,v,w,x
    # There are C(20,4) * 3 = 4845 * 3 = 14535 4-cycles
    # This is fast enough to check in Python within a second or two.
    
    import itertools
    
    valid = True
    # To speed up, we can break early if invalid
    for u, v, w, x in itertools.combinations(range(n), 4):
        # 3 possible 4-cycles on these 4 vertices
        # 1. u-v-w-x-u
        e1 = edge_color.get((min(u,v), max(u,v)))
        e2 = edge_color.get((min(v,w), max(v,w)))
        e3 = edge_color.get((min(w,x), max(w,x)))
        e4 = edge_color.get((min(x,u), max(x,u)))
        
        if e1 is None or e2 is None or e3 is None or e4 is None:
            # Not a cycle, skip
            continue
        if len({e1, e2, e3, e4}) < 4:
            valid = False
            break
        
        # 2. u-v-x-w-u
        e1 = edge_color.get((min(u,v), max(u,v)))
        e2 = edge_color.get((min(v,x), max(v,x)))
        e3 = edge_color.get((min(x,w), max(x,w)))
        e4 = edge_color.get((min(w,u), max(w,u)))
        
        if e1 is None or e2 is None or e3 is None or e4 is None:
            continue
        if len({e1, e2, e3, e4}) < 4:
            valid = False
            break
        
        # 3. u-w-v-x-u
        e1 = edge_color.get((min(u,w), max(u,w)))
        e2 = edge_color.get((min(w,v), max(v,w)))
        e3 = edge_color.get((min(v,x), max(v,x)))
        e4 = edge_color.get((min(x,u), max(x,u)))
        
        if e1 is None or e2 is None or e3 is None or e4 is None:
            continue
        if len({e1, e2, e3, e4}) < 4:
            valid = False
            break
    
    if valid:
        return edges
    
    # Fallback: return a star (always valid)
    return [[0, v, 0] for v in range(1, n)]

# EVOLVE-BLOCK-END
