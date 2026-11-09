"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Greenwood-Gleason style construction extended to n=50
    # Uses GF(25) for two 25-vertex components, joined by a bipartite color class.
    
    # GF(25) arithmetic setup
    # Primitive polynomial x^2 + x + 3 over GF(5)
    # Elements are pairs (a, b) representing a + b*x where x^2 = -x - 3 = 2x + 2
    # Multiplication: (a1, b1) * (a2, b2) = (a1a2 + 2b1b2, a1b2 + a2b1 + 2b1b2) mod 5
    
    def gf25_mul(p, q):
        a1, b1 = p
        a2, b2 = q
        a = (a1 * a2 + 2 * b1 * b2) % 5
        b = (a1 * b2 + a2 * b1 + 2 * b1 * b2) % 5
        return (a, b)
    
    def gf25_add(p, q):
        return ((p[0] + q[0]) % 5, (p[1] + q[1]) % 5)
        
    def gf25_sub(p, q):
        return ((p[0] - q[0]) % 5, (p[1] - q[1]) % 5)
        
    # Generate all 25 elements
    elements = [(a, b) for a in range(5) for b in range(5)]
    # Map element to index for O(1) lookup
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Find generator of GF(25)^*
    # Order is 24. A generator g must have order 24.
    # Let's just find one.
    gen = None
    for e in elements:
        if e == (0, 0): continue
        # Check order
        x = e
        order = 1
        while x != (1, 0):
            x = gf25_mul(x, e)
            order += 1
            if order > 24: break
        if order == 24:
            gen = e
            break
            
    if gen is None:
        # Fallback, though unlikely
        gen = (1, 1)

    # Generate cubic residues (subgroup of order 8)
    # H = { gen^(3k) for k in 0..7 }
    H = set()
    x = (1, 0)
    for _ in range(8):
        H.add(x)
        x = gf25_mul(x, gen)
        
    # Cosets
    # C0 = H
    # C1 = gen * H
    # C2 = gen^2 * H
    C1 = set(gf25_mul(gen, h) for h in H)
    C2 = set(gf25_mul(gf25_mul(gen, gen), h) for h in H)
    
    # Precompute color for difference d
    # color(d) = 0 if d in C0, 1 if d in C1, 2 if d in C2
    color_map = {}
    for e in elements:
        if e == (0, 0):
            color_map[e] = -1
        elif e in H:
            color_map[e] = 0
        elif e in C1:
            color_map[e] = 1
        else:
            color_map[e] = 2
            
    # Construct coloring for one K25
    def get_k25_colors():
        colors = []
        for i in range(25):
            for j in range(i + 1, 25):
                d = gf25_sub(elements[j], elements[i])
                colors.append(color_map[d])
        return colors

    colors1 = get_k25_colors()
    colors2 = get_k25_colors()
    
    # Combine
    # Vertices 0..24 are copy 1, 25..49 are copy 2
    # Edges within 0..24: colors1
    # Edges within 25..49: colors2
    # Edges between: color 3
    
    all_colors = []
    
    # Add colors for first block
    all_colors.extend(colors1)
    
    # Add edges between block 1 and block 2
    # i in 0..24, j in 25..49
    # Lexicographic order: (i, j) with i < j
    # The sequence of edges in K50 is:
    # (0,1), (0,2), ..., (0,24) -> 24 edges (within block 1)
    # (0,25), ..., (0,49) -> 25 edges (between)
    # (1,2), ..., (1,24) -> 23 edges (within block 1)
    # (1,25), ..., (1,49) -> 25 edges (between)
    # ...
    
    # It is easier to generate the full list in lexicographic order directly
    all_colors = []
    for i in range(50):
        for j in range(i + 1, 50):
            if j < 25:
                # Within block 1
                # Index in colors1:
                # Number of edges before i in K25: i*24 - i*(i-1)/2 ?
                # Edges (k, l) with k < l.
                # For a fixed i, edges (i, j) with j > i.
                # In the flat array of K25, the index for (i, j) is:
                # sum_{k=0}^{i-1} (24-k) + (j - i - 1)
                # sum_{k=0}^{i-1} (24-k) = 24*i - i*(i-1)/2
                idx = 24 * i - i * (i - 1) // 2 + (j - i - 1)
                all_colors.append(colors1[idx])
            elif i < 25:
                # Between blocks
                all_colors.append(3)
            else:
                # Within block 2
                # Map i, j to i', j' in 0..24
                i_prime = i - 25
                j_prime = j - 25
                idx = 24 * i_prime - i_prime * (i_prime - 1) // 2 + (j_prime - i_prime - 1)
                all_colors.append(colors2[idx])
                
    return {'n': 50, 'colors': all_colors}
# EVOLVE-BLOCK-END
