import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Try to construct n=50 or n=49
    # The record is n=50.
    # A known construction for n=50 uses the Paley graph of order 49 + 1 vertex?
    # Or it is a specific design.
    
    # Let's first build the n=49 solution using GF(49) quartic residues.
    # If that works, we have score 49/32 = 1.53.
    # Then we try to extend to 50.
    
    # GF(49) construction
    # 49 = 7^2
    # Irreducible polynomial: x^2 + x + 2 (discriminant 1-8 = -7 = 0 mod 7? No, -7 is 0 mod 7. Bad.)
    # Try x^2 + x + 3? Disc 1-12 = -11 = 5 mod 7. 5 is not a square mod 7 (1,4,2). So irreducible.
    # Let alpha be root of x^2 + x + 3 = 0 in GF(49).
    # alpha^2 = -alpha - 3 = 6alpha + 4.
    
    def mul(a, b):
        # a, b are in GF(49), represented as (a0, a1) where a = a0 + a1*alpha
        # a0, a1 in 0..6
        if isinstance(a, int): a = (a, 0)
        if isinstance(b, int): b = (b, 0)
        a0, a1 = a
        b0, b1 = b
        # (a0 + a1*a)(b0 + b1*a) = a0b0 + (a0b1 + a1b0)a + a1b1*a^2
        # a^2 = 6a + 4
        c0 = (a0*b0 + a1*b1*4) % 7
        c1 = (a0*b1 + a1*b0 + a1*b1*6) % 7
        return (c0, c1)
        
    def add(a, b):
        if isinstance(a, int): a = (a, 0)
        if isinstance(b, int): b = (b, 0)
        return ((a[0]+b[0])%7, (a[1]+b[1])%7)
        
    def neg(a):
        if isinstance(a, int): a = (a, 0)
        return ((-a[0])%7, (-a[1])%7)
        
    def sub(a, b):
        return add(a, neg(b))
        
    # Find primitive element
    # Order of GF(49)^* is 48.
    # Try (1, 1)
    x = (1, 1)
    order = 1
    while x != (1, 0):
        x = mul(x, (1, 1))
        order += 1
    # order should be 48
    
    # Generate powers
    powers = []
    x = (1, 0)
    for i in range(48):
        powers.append(x)
        x = mul(x, (1, 1))
        
    # Map element to index
    elem_to_idx = {p: i for i, p in enumerate(powers)}
    
    # Quartic residues are powers 0, 4, 8, ..., 44
    # Cosets:
    # C0: {4k}
    # C1: {4k+1}
    # C2: {4k+2}
    # C3: {4k+3}
    
    colors = []
    n = 49
    for i in range(n):
        for j in range(i+1, n):
            diff = sub(powers[i], powers[j])
            if diff == (0, 0):
                c = 0 # Should not happen for i!=j
            else:
                idx = elem_to_idx[diff]
                c = idx % 4
            colors.append(c)
            
    # Verify triangle-free
    # This is O(n^3) = 117,649 checks. Fast.
    def is_valid(colors, n):
        # Build adjacency list for each color
        adj = [set() for _ in range(4)]
        # Edge (i,j) has color c.
        # For each color c, check if there is a triangle.
        # i < j < k
        # (i,j) in c, (j,k) in c, (i,k) in c
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        for c in range(4):
            edges = adj[c]
            # Check triangles
            # For each edge (i,j), check if there is a k such that (i,k) and (j,k) in edges
            # This is slow if done naively.
            # Better: for each i, let N(i) be neighbors.
            # Check if any two neighbors of i are connected.
            neighbors = [set() for _ in range(n)]
            for (u, v) in edges:
                neighbors[u].add(v)
                neighbors[v].add(u)
            for i in range(n):
                nb = list(neighbors[i])
                for a in range(len(nb)):
                    for b in range(a+1, len(nb)):
                        if nb[b] in neighbors[nb[a]]:
                            return False
        return True

    if not is_valid(colors, 49):
        # Fallback to n=32 if 49 fails (should not)
        return {'n': 32, 'colors': []} # Placeholder
        
    # Try to extend to n=50
    # Add vertex 49.
    # We need to assign colors to edges (49, j) for j in 0..48.
    # Let's try a random assignment and check validity.
    # Or use a known pattern.
    # A common extension is to use the "infinite" vertex of the projective plane?
    
    # Let's try a simple local search on the 50th vertex colors.
    # But 50 vertices means 1225 edges.
    # The 49 edges from 49 to others.
    # We can try all 4^49 possibilities? No.
    # We can try random restarts.
    
    best_colors = colors
    best_n = 49
    
    # Try to find a valid coloring for n=50
    # We fix the first 49 vertices.
    # We vary the colors of edges (49, j).
    # There are 49 such edges.
    # We can use a SAT solver or local search.
    # Given time, let's try a few random assignments.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        new_colors = colors[:]
        # Randomly assign colors to edges from 49
        # Note: lexicographic order means edges (49, j) are at the end?
        # i < j.
        # Edges involving 49: (i, 49) for i=0..48.
        # In the list, they appear when j=49.
        # Indices:
        # For i in 0..48, j=49.
        # The index in the flat list:
        # Sum of (49-i) for i=0..48?
        # Total edges for n=49 is 49*48/2 = 1176.
        # Edges (i, 49) are the last 49 edges.
        # Indices 1176 to 1224.
        
        for k in range(49):
            new_colors[1176 + k] = random.randint(0, 3)
            
        if is_valid(new_colors, 50):
            return {'n': 50, 'colors': new_colors}
            
    return {'n': best_n, 'colors': best_colors}
