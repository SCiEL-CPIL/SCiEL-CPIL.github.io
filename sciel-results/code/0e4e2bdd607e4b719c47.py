def would_create_cycle(v, c, coloring):
    # Group neighbors by color
    neighbors_by_color = {}
    for nb in adj[v]:
        if coloring[nb] != -1 and coloring[nb] != c:
            d = coloring[nb]
            if d not in neighbors_by_color:
                neighbors_by_color[d] = []
            neighbors_by_color[d].append(nb)
    
    for d, nbs in neighbors_by_color.items():
        if len(nbs) < 2:
            # Only one neighbor with color d, no cycle possible from this pair
            continue
        # Check if any two neighbors in nbs are connected in the (c,d) subgraph
        # Do BFS from nbs[0] in the (c,d) subgraph (excluding v)
        visited = set()
        queue = [nbs[0]]
        visited.add(nbs[0])
        for nb2 in nbs[1:]:
            if nb2 in visited:
                return True
        while queue:
            curr = queue.pop(0)
            for w in adj[curr]:
                if w == v or w in visited:
                    continue
                cw = coloring[w]
                if cw == c or cw == d:
                    visited.add(w)
                    if w in nbs:
                        return True
                    queue.append(w)
    return False
