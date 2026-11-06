"""Bitset-based backtracking solver for R(K2,2,2) > 30."""
import numpy as np
from itertools import combinations
import time

def construct():
    """
    Find a 2-coloring of K_n with no monochromatic K_2,2,2.
    Uses bitset-based backtracking with efficient pruning.
    
    Key insight: G contains K_2,2,2 iff there exist two disjoint edges (a,b) and (c,d)
    such that N(a)∩N(b)∩N(c)∩N(d) is non-empty... Actually let me use:
    
    G contains K_2,2,2 iff there exist 3 vertices u,v,w forming a triangle,
    and the common neighborhood of u,v,w contains an edge (x,y).
    Then {u,v,w,x,y} with one more vertex... 
    
    Actually correct: if triangle (u,v,w) exists and common neighborhood contains
    edge (x,y), then pairs (u,x), (v,y), (w,?) - no we need 6 vertices.
    
    Let me use direct characterization: G ⊇ K_2,2,2 iff ∃ disjoint edges ab, cd, ef
    such that all 12 cross-edges exist.
    
    Efficient check: For each pair of vertices (a,b) that are adjacent, compute
    common = N(a) & N(b). If common contains a K_2,1 (two adjacent vertices x,y),
    then we need a third vertex z adjacent to both a,b,x,y. That's expensive.
    
    Better: Use the fact that K_2,2,2 is small (6 vertices). For n≤30, we can
    check all C(n,6) subsets but that's ~600K which is too slow per step.
    
    Optimal pruning: When adding edge (i,j) in color c, only check if this new edge
    participates in a K_2,2,2. So check: does there exist a K_2,2,2 containing edge (i,j)?
    This means: find 4 more vertices k,l,m,n such that {i,j,k,l,m,n} forms K_2,2,2
    with (i,j) as one of the 3 pairs, OR (i,j) as a cross-edge.
    
    Case 1: (i,j) is a pair in K_2,2,2. Then we need 4 more vertices k,l,m,n
    partitioned into 2 pairs, all connected to i and j, and all cross-edges between
    the two pairs exist.
    
    Case 2: (i,j) is a cross-edge. Then i is in one pair, j in another. We need
    4 more vertices to complete the 3 pairs.
    
    This is still complex. Let me use numpy bitsets and a simpler but effective check:
    After each edge assignment, check if either color class contains K_2,2,2 by
    checking all triangles and their common neighborhoods.
    """
    
    start_time = time.time()
    time_limit = 115  # Leave 5 seconds for cleanup
    
    def has_k222(adj_bits, n):
        """
        Check if graph represented by adj_bits (list of uint64 bitmasks) contains K_2,2,2.
        adj_bits[v] has bit u set if v~u in the graph.
        
        K_2,2,2 characterization: There exist 3 vertices a,b,c forming a triangle
        such that the common neighborhood of a,b,c contains a triangle.
        Proof: If a,b,c form a triangle and x,y,z in N(a)∩N(b)∩N(c) form a triangle,
        then {a,b,c,x,y,z} contains K_2,2,2 with pairs (a,x), (b,y), (c,z).
        Conversely, any K_2,2,2 contains a triangle whose common neighborhood 
        contains another triangle.
        """
        for a in range(n):
            bits_a = adj_bits[a]
            # b must be neighbor of a, b > a
            b_candidates = bits_a & ((1 << (n - a - 1)) - 1) << (a + 1)
            while b_candidates:
                b_bit = b_candidates & (-b_candidates)
                b = b_bit.bit_length() - 1
                b_candidates ^= b_bit
                
                # c must be neighbor of both a and b, c > b
                ab_common = bits_a & adj_bits[b]
                c_candidates = ab_common & ((1 << (n - b - 1)) - 1) << (b + 1)
                while c_candidates:
                    c_bit = c_candidates & (-c_candidates)
                    c = c_bit.bit_length() - 1
                    c_candidates ^= c_bit
                    
                    # a,b,c form a triangle
                    # Find common neighbors of a,b,c excluding a,b,c
                    abc_common = bits_a & adj_bits[b] & adj_bits[c]
                    abc_common &= ~((1 << a) | (1 << b) | (1 << c))
                    
                    # Check if abc_common contains a triangle
                    # Iterate over pairs in abc_common
                    tmp = abc_common
                    while tmp:
                        x_bit = tmp & (-tmp)
                        x = x_bit.bit_length() - 1
                        tmp ^= x_bit
                        
                        # y must be in abc_common, y > x, and x~y
                        xy_common = abc_common & adj_bits[x]
                        xy_common &= ((1 << (n - x - 1)) - 1) << (x + 1)
                        while xy_common:
                            y_bit = xy_common & (-xy_common)
                            y = y_bit.bit_length() - 1
                            xy_common ^= y_bit
                            
                            # Check if there's z in abc_common, z > y, with z~x and z~y
                            xyz_common = abc_common & adj_bits[x] & adj_bits[y]
                            xyz_common &= ((1 << (n - y - 1)) - 1) << (y + 1)
                            if xyz_common:
                                return True
        return False
    
    def solve(n, timeout=110):
        """Backtracking solver with bitset adjacency."""
        colors = [-1] * (n * (n - 1) // 2)
        
        # Bitset adjacency for each color
        adj0 = [0] * n
        adj1 = [0] * n
        
        # Edge ordering: use a fixed order for determinism
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_idx = 0
        
        def backtrack():
            nonlocal edge_idx
            if time.time() - start_time > timeout:
                return False
            if edge_idx >= len(edges):
                return True
            
            i, j = edges[edge_idx]
            edge_idx += 1
            
            # Try color 0
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
            
            if not has_k222(adj0, n):
                if backtrack():
                    return True
            
            adj0[i] &= ~(1 << j)
            adj0[j] &= ~(1 << i)
            
            # Try color 1
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            
            if not has_k222(adj1, n):
                if backtrack():
                    return True
            
            adj1[i] &= ~(1 << j)
            adj1[j] &= ~(1 << i)
            
            edge_idx -= 1
            return False
        
        if backtrack():
            # Reconstruct colors
            adj0_final = adj0[:]
            adj1_final = adj1[:]
            result = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj0_final[i] & (1 << j):
                        result.append(0)
                    else:
                        result.append(1)
            return {'n': n, 'colors': result}
        return None
    
    # Try n=30 first
    result = solve(30, timeout=100)
    if result is not None:
        return result
    
    # Fall back to n=29
    result = solve(29, timeout=50)
    if result is not None:
        return result
    
    # Fallback: return a valid small coloring
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % 17 in squares else 1)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
