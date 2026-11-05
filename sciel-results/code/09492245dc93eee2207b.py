    def is_triangle_free_set(edge_set, u, v):
        """Check if adding edge (u, v) to the graph represented by edge_set creates a triangle."""
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        # A triangle is created if there is a vertex w such that (u,w) and (v,w) are in edge_set.
        # We can check common neighbors.
        neighbors_u = {x for (a, b) in edge_set if a == u for x in [b] if x != v} | \
                      {x for (a, b) in edge_set if b == u for x in [a] if x != v}
        neighbors_v = {x for (a, b) in edge_set if a == v for x in [b] if x != u} | \
                      {x for (a, b) in edge_set if b == v for x in [a] if x != u}
        
        if neighbors_u & neighbors_v:
            return False
        return True
