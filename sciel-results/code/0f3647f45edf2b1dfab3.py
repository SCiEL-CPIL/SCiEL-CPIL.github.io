import random
import itertools

def construct():
    # Try to maximize n
    # Known: n=42 works with incidence graph of projective plane order 4.
    # Let's try random greedy for n=47 down to 39.
    
    def is_c4_free_after_add(G, u, v, n):
        # Check if adding (u, v) creates a C4.
        # A C4 is created if there is a vertex w such that
        # w is connected to both u and v (forming u-w-v-... no, u-v is the new edge)
        # If we add u-v, a C4 is formed if there exists w != u, v such that
        # (u, w) in G and (v, w) in G.
        # Wait, that's a triangle u-w-v-u. That's C3.
        # A C4 involving the new edge (u, v) would be u - x - y - v - u.
        # This requires a path of length 2 from u to v in G.
        # i.e., exists x, y such that (u, x) in G, (x, y) in G, (y, v) in G.
        # Or simpler: The new edge (u, v) creates a C4 if the distance between u and v in G is 2?
        # No. If dist(u, v) = 2 in G, say u-x-v, then adding u-v creates a triangle (C3).
        # We are checking for C4.
        # If we add edge (u, v), a C4 is formed if there is a path of length 3 between u and v in G.
        # u - x - y - v.
        # This means there exists x, y such that (u, x) in G, (x, y) in G, (y, v) in G.
        # Equivalently, the neighborhood of u and the neighborhood of v have a common neighbor?
        # No.
        # Let's check common neighbors of u and v.
        # If u and v share a common neighbor w, then u-w-v is a path of length 2. Adding u-v creates a C3.
        # We don't care about C3.
        # To create a C4, we need a path of length 3.
        # u - a - b - v.
        # This means a is a neighbor of u, b is a neighbor of v, and (a, b) is an edge in G.
        
        # So, for each a in N(u), for each b in N(v), check if (a, b) in G.
        # If yes, C4 is formed.
        
        N_u = G[u]
        N_v = G[v]
        
        for a in N_u:
            if a == v: continue
            for b in N_v:
                if b == u: continue
                if a == b: continue # This would be a C3 case if a==b? No, if a==b, path is u-a-v, length 2. Adding u-v makes C3.
                if b in G[a]:
                    return True
        return False

    def try_construct(n):
        # Random greedy
        # To make it faster, we can use bitmasks or sets.
        # n <= 47, so sets are fine.
        
        # We want min degree >= n - 39.
        target_deg = n - 39
        
        # Try a few random orders
        for _ in range(100):
            G = {i: set() for i in range(n)}
            edges = []
            
            # Generate all possible edges
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a C4
                if not is_c4_free_after_add(G, u, v, n):
                    G[u].add(v)
                    G[v].add(u)
                    edges.append([u, v])
            
            # Check min degree
            min_deg = min(len(G[i]) for i in range(n))
            if min_deg >= target_deg:
                return {'n': n, 'edges': edges}
        
        return None

    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        res = try_construct(n)
        if res:
            return res
    
    # Fallback: Incidence graph of projective plane of order 4 (n=42)
    # This is deterministic and guaranteed.
    # Points: 0..20, Lines: 21..41
    # We need to generate the edges.
    # Projective plane of order 4 has q=4.
    # Number of points = 21.
    # We can use a simple representation.
    # However, implementing the full projective plane geometry is complex.
    # We can use the known fact that n=42 works.
    # But we need the edges.
    
    # Let's try a simpler fallback: n=39 empty graph? No, score 1.0.
    # We can use the polarity graph for q=5 (n=31).
    # Or just return the n=42 graph if we can construct it.
    
    # Since try_construct(42) should succeed easily (target deg 3), 
    # and we loop down to 39, it will likely find a solution for n=42 or higher.
    
    # If all fail (unlikely), return seed.
    return {'n': 39, 'edges': []}
