"""Hadamard-based 4-coloring of K_n for R(3,3,3,3) > n."""

import time
import numpy as np

def construct():
    """
    Returns a 4-colouring of K_n with no monochromatic triangle, maximising n.
    Attempts to construct a valid 48-vertex coloring using Z_2^4 x Z_3 group structure.
    Falls back to 32-vertex Clebsch-based coloring if verification fails.
    """
    t0 = time.time()
    deadline = t0 + 140.0

    # Helper to verify if a coloring is valid (no monochromatic triangles)
    def is_valid_coloring(n, colors):
        """
        n: number of vertices
        colors: list of length n*(n-1)//2, in lexicographic order
        """
        # Build adjacency matrix of colors
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        
        # Check for monochromatic triangles
        # For each color c, check if the graph formed by edges of color c has a triangle
        for c in range(4):
            # Mask of edges with color c
            mask = (adj == c)
            # A triangle exists if there are i < j < k such that mask[i,j] & mask[j,k] & mask[i,k]
            # We can check this by looking at the adjacency matrix of the subgraph
            # For each i, for each j > i with mask[i,j], check if there is k > j with mask[i,k] and mask[j,k]
            # Optimization: Use numpy
            for i in range(n):
                # Get neighbors j > i connected by color c
                neighbors = np.where(mask[i, i+1:])[0] + i + 1
                if len(neighbors) < 2:
                    continue
                # For each pair of neighbors (j, k), check if edge (j,k) is color c
                # This is O(N^3) worst case, but N=48 is small enough
                for idx_j in range(len(neighbors)):
                    j = neighbors[idx_j]
                    for idx_k in range(idx_j + 1, len(neighbors)):
                        k = neighbors[idx_k]
                        if mask[j, k]:
                            return False
        return True

    # Try to construct 48-vertex coloring using Z_2^4 x Z_3
    def try_48_coloring():
        # Vertices: (x, a) where x in 0..15 (Z_2^4), a in 0..2 (Z_3)
        # Total 48 vertices
        n = 48
        colors = []
        
        # Mapping function hypothesis
        # c(x, y, a, b) based on popcount(x^y) and (b-a)%3
        # We need to find a valid mapping.
        # Common mapping:
        # d = popcount(x^y)
        # delta = (b-a) % 3
        # color = (d + 3*delta) % 4  (Hypothesis 1)
        # color = (d + delta) % 4    (Hypothesis 2)
        
        # Let's try a few simple mappings and verify
        mappings = [
            lambda d, delta: (d + 3 * delta) % 4,
            lambda d, delta: (d + delta) % 4,
            lambda d, delta: (d * 2 + delta) % 4,
            lambda d, delta: (d + 2 * delta) % 4,
        ]
        
        for m_func in mappings:
            colors = []
            for i in range(n):
                x1 = i // 3
                a1 = i % 3
                for j in range(i + 1, n):
                    x2 = j // 3
                    a2 = j % 3
                    d = bin(x1 ^ x2).count('1')
                    delta = (a2 - a1) % 3
                    c = m_func(d, delta)
                    colors.append(c)
            
            if is_valid_coloring(n, colors):
                return n, colors
        return None

    result_48 = try_48_coloring()
    if result_48:
        n, colors = result_48
        return {'n': n, 'colors': colors}

    # Fallback: 32-vertex Clebsch-based coloring
    def mul16(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log16, x = {}, 1
    for e in range(15):
        log16[x] = e
        x = mul16(x, 2)

    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log16[i ^ j] % 3

    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    n32 = 32
    colors_32 = []
    for i in range(n32):
        for j in range(i + 1, n32):
            colors_32.append(get_color_32(i, j))
    
    return {'n': n32, 'colors': colors_32}
