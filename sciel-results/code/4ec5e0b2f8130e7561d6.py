"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import heapq

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

    def induced_matching_coloring(edges, n, max_degree):
        """
        Color edges so that each color class is an induced matching.
        Uses a greedy approach with careful conflict detection.
        """
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, precompute the set of vertices that would conflict
        # An edge (u,v) conflicts with (x,y) if:
        # 1. They share a vertex (u==x, u==y, v==x, v==y), OR
        # 2. They are joined by an edge: u-x, u-y, v-x, v-y exist in adj
        # So the "forbidden neighborhood" for edge (u,v) includes:
        # - u, v themselves
        # - all neighbors of u and v
        # If another edge (x,y) has x or y in this set, they conflict.
        
        # Greedy coloring: assign each edge the smallest available color
        # where no previously colored edge with that color conflicts with it.
        
        # To make this more efficient and get fewer colors, we can use
        # a smarter ordering. Let's try ordering edges by their "degree" 
        # in the line graph (i.e., how many other edges they conflict with).
        
        # Build conflict graph implicitly
        edge_index = {}
        for i, e in enumerate(edges):
            edge_index[e] = i
        
        num_edges = len(edges)
        
        # For each edge, compute the set of vertices it "touches" including neighbors
        # An edge e=(u,v) conflicts with f=(x,y) if {x,y} intersects {u,v} ∪ N(u) ∪ N(v)
        edge_vertices = []
        for e in edges:
            u, v = e
            verts = set()
            verts.add(u)
            verts.add(v)
            verts.update(adj[u])
            verts.update(adj[v])
            edge_vertices.append(verts)
        
        # Order edges by number of conflicts (descending) - most constrained first
        # Count conflicts for each edge
        conflict_counts = [0] * num_edges
        for i in range(num_edges):
            vi = edge_vertices[i]
            for j in range(i+1, num_edges):
                vj = edge_vertices[j]
                if vi & vj:
                    conflict_counts[i] += 1
                    conflict_counts[j] += 1
        
        # Sort edges by conflict count descending, then by original index for stability
        order = sorted(range(num_edges), key=lambda i: (-conflict_counts[i], i))
        
        color_of_edge = [-1] * num_edges
        # For each color, track which edges have been assigned to it
        color_edges = {}  # color -> list of edge indices
        
        # For fast lookup, maintain for each color the union of vertex sets
        color_vertex_sets = {}  # color -> set of vertices
        
        for idx in order:
            e = edges[idx]
            vi = edge_vertices[idx]
            
            # Find the smallest color that doesn't conflict
            assigned = False
            for c in range(num_edges):  # Upper bound on colors needed
                if c not in color_edges:
                    color_edges[c] = []
                    color_vertex_sets[c] = set()
                
                # Check if edge idx conflicts with any edge in color c
                conflicts = False
                for other_idx in color_edges[c]:
                    if vi & edge_vertices[other_idx]:
                        conflicts = True
                        break
                
                if not conflicts:
                    color_of_edge[idx] = c
                    color_edges[c].append(idx)
                    color_vertex_sets[c] = color_vertex_sets[c] | vi
                    assigned = True
                    break
            
            if not assigned:
                # Should not happen, but just in case
                c = num_edges
                color_of_edge[idx] = c
                color_edges[c] = [idx]
                color_vertex_sets[c] = vi
        
        # Return colors in sorted edge order
        return [color_of_edge[i] for i in range(num_edges)]
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = induced_matching_coloring(edges, n, d)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
