"""Program for R(K2,2,2) maximization using bitmask constraint satisfaction."""
import numpy as np

def construct():
    # We want to maximize n such that there exists a 2-coloring of K_n
    # with no monochromatic K_{2,2,2}.
    
    # A monochromatic K_{2,2,2} in color c exists iff there are two disjoint
    # sets A, B of size 2 each, and a set C of size 2, such that all edges
    # between A and C, and between B and C, are color c.
    # Equivalently: for some pair (u,v) of vertices, and some pair (x,y) of vertices
    # with x,y not in {u,v}, if u~x, u~y, v~x, v~y are all color c, then we need
    # to check if there's another pair... 
    # 
    # Actually, K_{2,2,2} is the complete tripartite graph with parts of size 2.
    # So we need three disjoint pairs (a1,a2), (b1,b2), (c1,c2) such that all
    # edges between any two of the three pairs are the same color c.
    # 
    # A simpler characterization: A color-c K_{2,2,2} exists iff there exist
    # two vertices u, v such that their common color-c neighborhood contains
    # a color-c K_{2,2}. Because: if u,v have common neighbors x1,x2,y1,y2 in color c,
    # and the subgraph on {x1,x2,y1,y2} in color c contains K_{2,2}, say with parts
    # {x1,x2} and {y1,y2}, then {u,v}, {x1,x2}, {y1,y2} form a K_{2,2,2} in color c.
    # Wait, that's not quite right. K_{2,2,2} needs all 12 edges between the three pairs.
    # If u,v are connected to all of x1,x2,y1,y2 in color c, and {x1,x2} is connected
    # to {y1,y2} in color c, then we have K_{2,2,2}.
    #
    # So the condition is: for every pair (u,v), the common color-c neighborhood N_c(u) ∩ N_c(v)
    # must NOT contain a color-c K_{2,2}. A K_{2,2} in a set S exists iff there are two
    # disjoint pairs in S that are completely connected in color c. Equivalently, the
    # color-c subgraph induced on S does not contain K_{2,2}.
    #
    # For a set S, it contains K_{2,2} iff there exist a,b,c,d in S such that all four
    # cross edges ab... wait. K_{2,2} has parts {a,b} and {c,d}, with edges a-c, a-d, b-c, b-d.
    # So we need two vertices a,b and two vertices c,d (all distinct) such that all four
    # edges between {a,b} and {c,d} are color c.
    #
    # So: for color c, a set S avoids K_{2,2} iff for every partition of 4 distinct vertices
    # in S into two pairs {a,b} and {c,d}, at least one of the four cross edges is not color c.
    #
    # This is getting complex. Let me use a different approach:
    # Use a known construction or search for small n.
    #
    # The record is 29. Let me try to construct a 29-vertex coloring.
    # 
    # One approach: use the Paley graph on 29 vertices? 29 is prime, 29 ≡ 1 mod 4.
    # Squares mod 29: let's compute.
    # Actually, let me try a random search with constraint checking for n=29, 30.
    
    def has_k222(colors, n, color):
        """Check if color class contains K_{2,2,2}."""
        # Build adjacency for the given color
        # colors is a list of length n*(n-1)//2 in lexicographic order
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        # Check for K_{2,2,2}: three disjoint pairs with all cross edges present
        # For efficiency, check: for each pair (u,v), compute common neighbors
        # Then check if common neighbors contain K_{2,2}
        
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] == 0:
                    continue
                # Common neighbors of u and v
                common = []
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        common.append(w)
                # Check if common contains K_{2,2}
                # K_{2,2} in common: two pairs {a,b}, {c,d} with all 4 cross edges
                m = len(common)
                if m < 4:
                    continue
                # For each pair of pairs in common
                found = False
                for a_idx in range(m):
                    a = common[a_idx]
                    for b_idx in range(a_idx+1, m):
                        b = common[b_idx]
                        if not adj[a][b]:
                            continue
                        # Now check if there exist c,d in common \ {a,b} with all cross edges
                        for c_idx in range(m):
                            c = common[c_idx]
                            if c == a or c == b:
                                continue
                            for d_idx in range(c_idx+1, m):
                                d = common[d_idx]
                                if d == a or d == b:
                                    continue
                                if (adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d]):
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    return True
        return False
    
    def check_valid(colors, n):
        """Check if both color classes are K_{2,2,2}-free."""
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)
    
    # Try to find a valid coloring for increasing n
    # Start from a known good construction or random search
    
    # Let's try the Paley graph on primes p where p ≡ 1 mod 4
    # For p=29: squares mod 29
    for p in [29, 31]:
        if p <= 31:
            squares = {x*x % p for x in range(1, p)}
            squares.discard(0)
            colors = [0 if (j-i) % p in squares else 1 for i in range(p) for j in range(i+1, p)]
            if check_valid(colors, p):
                return {'n': p, 'colors': colors}
    
    # Try random search for n=29
    import random
    random.seed(42)
    
    for n in [29, 30]:
        for attempt in range(10000):
            colors = [random.choice([0, 1]) for _ in range(n*(n-1)//2)]
            if check_valid(colors, n):
                return {'n': n, 'colors': colors}
    
    # Fallback to n=17 Paley
    squares = {x*x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j-i) % 17 in squares else 1 for i in range(17) for j in range(i+1, 17)]}
