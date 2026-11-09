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
        
        # Build adjacency list for the original graph
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Calculate conflict for each edge
        # Two edges (u,v) and (x,y) conflict if:
        # 1. They share a vertex (u==x, u==y, v==x, v==y)
        # 2. They are connected by an edge (v==x and (u,y) in adj, etc.)
        # It's faster to precompute the set of vertices "near" an edge.
        # near(e) = {u, v} U N(u) U N(v)
        # Two edges e1, e2 conflict if set(e1) & near(e2) is non-empty?
        # Actually, if e1=(u,v) and e2=(x,y), they conflict if:
        # - u==x or u==y or v==x or v==y (share vertex)
        # - (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E (joined by edge)
        
        # Let's define conflict index for each edge i
        m = len(edges)
        # conflict_mask[i] is a bitset or set of indices j that conflict with i
        # Since m is small (max ~240), we can use sets or bitsets.
        # Using sets of indices is simple.
        
        # Precompute near vertices for each edge
        edge_vertices = [set(e) for e in edges]
        near_vertices = []
        for e in edges:
            u, v = e
            near = set([u, v])
            near.update(adj.get(u, set()))
            near.update(adj.get(v, set()))
            near_vertices.append(near)
        
        # Determine conflicts
        # conflict[i] is a set of j such that i and j cannot have the same color
        conflicts = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            near_i = near_vertices[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                # Check if they share a vertex or are joined
                # If any vertex of j is in near_i, they conflict?
                # Let's verify:
                # If uj is in near_i, then uj is either ui, vi, or neighbor of ui/vi.
                # If uj == ui or uj == vi -> share vertex.
                # If uj is neighbor of ui -> edge (ui, uj) exists. Since (uj, vj) is an edge, 
                #   then ui and vj are connected via uj? No, the condition is "joined by an edge".
                #   The pair of edges (ui, vi) and (uj, vj) are joined by an edge if there is an edge
                #   connecting a vertex of the first to a vertex of the second.
                #   So if (ui, uj) is an edge, then the two edges are joined.
                #   So yes, if uj is in near_i, they conflict.
                if uj in near_i or vj in near_i:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Sort edges by conflict degree descending
        order = sorted(range(m), key=lambda i: len(conflicts[i]), reverse=True)
        
        color = [0] * m
        used_colors = [set() for _ in range(m)] # used_colors[i] is the set of colors used by neighbors of i? No.
        # We need to track which colors are used by conflicting edges.
        # Let's maintain a list of sets: color_classes[c] = set of edge indices with color c
        # Or simply, for each edge, check which colors are taken by its conflicts.
        
        # Efficient check:
        # For edge i, taken_colors = { color[j] for j in conflicts[i] }
        
        max_color = 0
        for i in order:
            taken = set()
            for j in conflicts[i]:
                taken.add(color[j])
            c = 0
            while c in taken:
                c += 1
            color[i] = c
            if c > max_color:
                max_color = c
        
        # Return colors in original edge order
        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() - start_time > 140:
            # Fallback to greedy if running out of time, though this loop is fast
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
