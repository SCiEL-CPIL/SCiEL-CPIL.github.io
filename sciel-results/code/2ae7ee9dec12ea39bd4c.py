    def can_add(neighbors, u, v):
        common_uv = neighbors[u] & neighbors[v]
        if common_uv.bit_count() > 2:
            return False
            
        # Check constraints induced on (u, x) and (v, x) for x in common_uv
        # If x is in common_uv, x is neighbor of u and v.
        # Adding (u,v) adds v to N(u) and u to N(v).
        # For pair (u, x):
        #   New common neighbors = Old common neighbors + (v if v in N(x) else 0).
        #   v is in N(x) because x is in N(v) (undirected).
        #   So common(u, x) increases by 1.
        #   We need new common <= 2 => old common <= 1.
        #   So check: (neighbors[u] & neighbors[x]).bit_count() <= 1.
        
        for x in range(n):
            if common_uv & (1 << x):
                if (neighbors[u] & neighbors[x]).bit_count() > 1:
                    return False
                if (neighbors[v] & neighbors[x]).bit_count() > 1:
                    return False
        return True
    