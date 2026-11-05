"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def color_induced_matching(edges, n):
        """Color edges so each color class is an induced matching."""
        if not edges:
            return []
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, compute the set of vertices that are "forbidden"
        # Edge (u,v) conflicts with edge (x,y) if they share a vertex or
        # there's an edge between {u,v} and {x,y}
        # So edge (u,v) is compatible with edge (x,y) iff
        # {x,y} is disjoint from N[u] ∪ N[v] (closed neighborhood)
        
        edge_list = list(edges)
        m = len(edge_list)
        
        # Precompute closed neighborhood for each vertex
        closed_neigh = [set(adj[i]) | {i} for i in range(n)]
        
        # For each edge, compute the set of vertices it "covers" (conflicts with)
        # Edge i=(u,v) conflicts with edge j=(x,y) if {x,y} intersects (closed_neigh[u] | closed_neigh[v])
        edge_forbidden_verts = []
        for u, v in edge_list:
            forbidden = closed_neigh[u] | closed_neigh[v]
            edge_forbidden_verts.append(forbidden)
        
        # Build conflict: edge i and edge j conflict if
        # edge_j's vertices intersect edge_i's forbidden set
        # i.e., for edge j=(x,y), if x in edge_forbidden_verts[i] or y in edge_forbidden_verts[i]
        
        remaining = set(range(m))
        colors = [None] * m
        next_color = 0
        
        while remaining:
            if time.time() > deadline:
                # Fallback: assign unique colors to remaining
                for idx in remaining:
                    colors[idx] = next_color
                    next_color += 1
                break
            
            # Find a large independent set in the conflict graph induced by remaining
            # Greedy: sort by degree in conflict subgraph (ascending), pick greedily
            # Compute degree of each remaining edge in the conflict subgraph
            
            # For efficiency, use a simpler heuristic:
            # Pick edges in order of minimum "conflict degree" among remaining
            # An edge's conflict degree = number of other remaining edges it conflicts with
            
            # Let's use a different approach: greedy set packing
            # Sort remaining edges by some priority, then greedily select non-conflicting ones
            
            # Priority: edges with fewer conflicts (lower degree) are easier to place
            # Compute degree for each remaining edge
            deg = {}
            for i in remaining:
                fi = edge_forbidden_verts[i]
                d_i = 0
                for j in remaining:
                    if j == i:
                        continue
                    uj, vj = edge_list[j]
                    if uj in fi or vj in fi:
                        d_i += 1
                deg[i] = d_i
            
            # Sort by degree ascending, then by edge index for determinism
            sorted_remaining = sorted(remaining, key=lambda x: (deg[x], x))
            
            # Greedily select non-conflicting edges
            selected = []
            selected_forbidden = set()
            for i in sorted_remaining:
                if i not in remaining:
                    continue
                ui, vi = edge_list[i]
                # Check if this edge conflicts with any already selected
                conflict = False
                for j in selected:
                    uj, vj = edge_list[j]
                    if uj in edge_forbidden_verts[i] or vj in edge_forbidden_verts[i]:
                        conflict = True
                        break
                if not conflict:
                    selected.append(i)
                    selected_forbidden |= edge_forbidden_verts[i]
            
            # Assign color
            for i in selected:
                colors[i] = next_color
                remaining.discard(i)
            next_color += 1
        
        return colors
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = color_induced_matching(edges, n)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
