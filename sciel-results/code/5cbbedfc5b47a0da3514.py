"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    start_time = time.time()
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def color_graph(edges):
        if not edges:
            return []
        
        m = len(edges)
        
        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Precompute conflict sets using bitmasks for efficiency
        # Two edges conflict if they share a vertex or are joined by an edge
        # i.e., for edges (u,v) and (x,y), conflict if any of u,v is in N[x]∪N[y] or vice versa
        # More precisely: conflict if (u==x or u==y or v==x or v==y) or 
        #                      ((u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E)
        
        # For each edge i, compute bitmask of conflicting edges
        # Use integer bitmasks since m <= ~240 (fits in Python int)
        
        # Precompute: for each edge i, the set of vertices "covered" by it (endpoints + their neighbors)
        # Edge i and j conflict iff edge j has at least one endpoint in cover(i)
        # Wait, that's not quite right. Let me think again.
        # Edge i = (u, v). Cover(i) = {u, v} ∪ N(u) ∪ N(v)
        # Edge j = (x, y). They conflict if:
        #   - x or y is in {u, v} (share vertex), OR
        #   - x or y is in N(u) or N(v) (joined by an edge)
        # So yes: conflict iff x ∈ cover(i) or y ∈ cover(i)
        
        covers = []
        for u, v in edges:
            cov = {u, v}
            cov.update(adj.get(u, set()))
            cov.update(adj.get(v, set()))
            covers.append(cov)
        
        # Build conflict bitmasks
        conflict_masks = [0] * m
        for i in range(m):
            cov_i = covers[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov_i or v_j in cov_i:
                    mask |= (1 << j)
            conflict_masks[i] = mask
        
        # Greedy coloring with multiple orderings, keep best
        best_color = None
        best_max_color = m + 1
        
        # Generate several orderings
        orderings = []
        
        # 1. Descending conflict degree
        orderings.append(sorted(range(m), key=lambda i: bin(conflict_masks[i]).count('1'), reverse=True))
        
        # 2. Ascending conflict degree
        orderings.append(sorted(range(m), key=lambda i: bin(conflict_masks[i]).count('1')))
        
        # 3. Random orderings
        rng = Random(42)
        for _ in range(20):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)
        
        # 4. Degree-based ordering (by endpoint degrees)
        orderings.append(sorted(range(m), key=lambda i: len(adj.get(edges[i][0], set())) + len(adj.get(edges[i][1], set())), reverse=True))
        
        for order in orderings:
            color = [-1] * m
            # Track which colors are used by conflicts for each edge
            # For efficiency, maintain color_used_by: for each color c, bitmask of edges with color c
            # Then for edge i, taken colors = {c : conflict_masks[i] & color_used_by[c] != 0}
            
            max_color = 0
            color_used_by = []  # list of bitmasks, one per color
            
            for i in order:
                taken = set()
                cm = conflict_masks[i]
                for c, mask in enumerate(color_used_by):
                    if cm & mask:
                        taken.add(c)
                
                c = 0
                while c in taken:
                    c += 1
                
                color[i] = c
                if c >= len(color_used_by):
                    color_used_by.append(0)
                color_used_by[c] |= (1 << i)
                if c > max_color:
                    max_color = c
            
            if max_color < best_max_color:
                best_max_color = max_color
                best_color = color[:]
        
        return best_color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() - start_time > 130:
            # Fallback to simple greedy
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
            
        edges = instance(*spec)
        colors = color_graph(edges)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
