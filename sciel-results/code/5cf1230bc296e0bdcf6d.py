<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Eight disjoint copies of K5 (a K5 has no six-cycle).
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    return edges
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    n = 40
    
    # Start with 8 disjoint K5s (80 edges, no C6)
    adj = [[False] * n for _ in range(n)]
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    def would_create_c6(u, v):
        """Check if adding edge (u,v) creates a C6."""
        # A C6 through (u,v) means there's a path of length 4 from u to v
        # Path: u - a - b - c - d - v where a,b,c,d are distinct
        # This means there exists a path of length 3 from u to v in the graph without using the edge (u,v)
        # Actually, adding (u,v) creates a C6 iff there's a path of length 4 from u to v
        # i.e., u - x - y - z - w - v where x,y,z,w are all distinct and not u or v
        
        # More efficiently: check if there's a path of length 4 from u to v
        # This is equivalent to: exists x in N(u), y in N(x)\{u}, z in N(y)\{x}, w in N(z)\{y} such that w in N(v) and all distinct
        
        # Let's use a more efficient check:
        # For each neighbor x of u (excluding v if it were already there, but it's not):
        #   For each neighbor y of x (excluding u):
        #     For each neighbor z of y (excluding x):
        #       If z is a neighbor of v and z != u, we need to check distinctness
        #       Actually we need a path u-x-y-z-w-v where w is neighbor of v
        #       So: for each x in N(u), for each y in N(x) \ {u}, for each z in N(y) \ {x},
        #         if z in N(v) and z != u and all of x,y,z distinct from each other and from u,v
        #         then we have a C6 (u-x-y-z-v is length 4, plus edge u-v makes it a cycle of length 5... wait)
        
        # Wait, let me reconsider. A 6-cycle is u1-u2-u3-u4-u5-u6-u1.
        # If we add edge (u,v), a C6 containing (u,v) would be: u - a - b - c - d - v - u
        # So we need a path of length 4 from u to v: u-a-b-c-d-v
        # This means: a in N(u), b in N(a)\{u}, c in N(b)\{a}, d in N(c)\{b}, and d in N(v)
        # with all vertices distinct.
        
        # Efficient check:
        # For each a in N(u):
        #   For each b in N(a) \ {u, v}:
        #     For each c in N(b) \ {a, u, v}:
        #       For each d in N(c) \ {b, u, v, a}:
        #         if d in N(v) and d not in {a, b, c, u}:
        #           return True
        
        # This could be expensive. Let's optimize.
        
        nu = [x for x in range(n) if adj[u][x] and x != v]
        for a in nu:
            na = [x for x in range(n) if adj[a][x] and x != u and x != v]
            for b in na:
                nb = [x for x in range(n) if adj[b][x] and x != a and x != u and x != v]
                for c in nb:
                    nc = [x for x in range(n) if adj[c][x] and x != b and x != u and x != v]
                    for d in nc:
                        if d in adj[v] and d != a and d != b and d != c:
                            return True
        return False
    
    # Try to add edges greedily with randomization
    all_possible = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                all_possible.append((i, j))
    
    random.shuffle(all_possible)
    
    # Multiple passes to try adding edges
    for pass_num in range(20):
        if time.time() > deadline:
            break
        random.shuffle(all_possible)
        for (u, v) in all_possible:
            if time.time() > deadline:
                break
            if adj[u][v]:
                continue
            if not would_create_c6(u, v):
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
    
    return best_edges
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
