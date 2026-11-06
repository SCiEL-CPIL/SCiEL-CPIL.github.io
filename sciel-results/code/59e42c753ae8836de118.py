def check_non_2_colorable(n, edges):
    if not edges: return False
    edge_masks = []
    for e in edges:
        m = 0
        for v in e: m |= 1 << v
        edge_masks.append(m)
        
    # Precompute incidence
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    colors = [-1] * n
    # counts[i] = [count0, count1]
    counts = [[0,0] for _ in range(len(edges))]
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                c0, c1 = counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    # Must be 1
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            counts[i][1] += 1
                            changed = True
                            break
                elif c1 == 4 and c0 == 0:
                    # Must be 0
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            counts[i][0] += 1
                            changed = True
                            break
        return True

    def backtrack():
        if not propagate():
            return False
            
        # Find uncolored vertex
        # Heuristic: Pick vertex in an edge with highest constraint tightness
        best_v = -1
        best_score = -1
        for v in range(n):
            if colors[v] == -1:
                # Score based on degree or constraint tightness
                # Simple: just pick first? No, pick one in a "tight" edge
                for i in vertex_edges[v]:
                    c0, c1 = counts[i]
                    if c0 + c1 > 2: # At least 2 colored in this edge
                        score = c0 + c1
                        if score > best_score:
                            best_score = score
                            best_v = v
                if best_v != -1:
                    break # Just pick the first good one? Or scan all?
        
        if best_v == -1:
            # All colored? Check if valid
            # If all colored and propagate didn't fail, it's a valid coloring
            return True
            
        # Try 0
        colors[best_v] = 0
        saved_counts = []
        for i in vertex_edges[best_v]:
            saved_counts.append((counts[i][0], counts[i][1]))
            counts[i][0] += 1
        if backtrack():
            return True
            
        colors[best_v] = -1
        for i, (c0, c1) in zip(vertex_edges[best_v], saved_counts):
            counts[i] = [c0, c1]
            
        # Try 1
        colors[best_v] = 1
        saved_counts = []
        for i in vertex_edges[best_v]:
            saved_counts.append((counts[i][0], counts[i][1]))
            counts[i][1] += 1
        if backtrack():
            return True
            
        colors[best_v] = -1
        for i, (c0, c1) in zip(vertex_edges[best_v], saved_counts):
            counts[i] = [c0, c1]
            
        return False

    return not backtrack()
