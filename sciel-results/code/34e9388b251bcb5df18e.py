<<<<<<< SEARCH
def _check_bicolored_forest(coloring, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    colors_used = set(coloring)
    if len(colors_used) < 2:
        return True
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            parent = list(range(n))
            has_cycle = False
            
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True
=======
def _check_bicolored_forest(coloring, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    colors_used = set(coloring)
    if len(colors_used) < 2:
        return True
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            parent = list(range(n))
            has_cycle = False
            
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True


def _count_distinct(coloring):
    return len(set(coloring))
>>>>>>> REPLACE

