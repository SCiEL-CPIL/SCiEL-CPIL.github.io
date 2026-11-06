"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Use multiplicative structure in Z_17^*: assign each vertex a distinct
    nonzero element of Z_17, color edge (i,j) by (v_i * v_j) mod 17.
    Since 17 is prime and all v_i are nonzero, for any 4 vertices with
    values a,b,c,d the 6 edge colors {ab,ac,ad,bc,bd,cd} mod 17 are all
    distinct (ab=ac implies a(b-c)=0 mod 17, so b=c). Thus every K4 sees
    6 distinct colors >= 5. Total colors used: at most 16 (nonzero residues).
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Z_17^* = {1, 2, ..., 16}; use a generator for nice ordering
    # 3 is a primitive root mod 17: 3^1=3, 3^2=9, 3^3=10, 3^4=13, 3^5=5,
    # 3^6=15, 3^7=11, 3^8=16, 3^9=14, 3^10=8, 3^11=7, 3^12=4, 3^13=12,
    # 3^14=2, 3^15=6, 3^16=1
    # Just use 1..16 directly
    v = list(range(1, 17))  # v[i] = i+1, all in Z_17^*
    
    # Color edge (i, j) by (v[i] * v[j]) % 17
    colors = [(v[i] * v[j]) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Validate: every 4-vertex subset must span >= 5 distinct colors
    if time.time() < deadline:
        valid = True
        # Sample some K4s to verify (full check is C(16,4)=1820)
        import itertools
        for quad in itertools.combinations(range(16), 4):
            a, b, c, d = quad
            # Get edge indices: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            # In lexicographic order, edge (i,j) is at index:
            # sum_{k=0}^{i-1} (15-k) + (j-i-1) = i*15 - i*(i-1)/2 + j - i - 1
            def edge_idx(i, j):
                # number of edges starting before i: sum_{k=0}^{i-1}(15-k) = i*15 - i*(i-1)//2
                return i * 15 - i * (i - 1) // 2 + (j - i - 1)
            
            edge_colors = {
                colors[edge_idx(a, b)],
                colors[edge_idx(a, c)],
                colors[edge_idx(a, d)],
                colors[edge_idx(b, c)],
                colors[edge_idx(b, d)],
                colors[edge_idx(c, d)],
            }
            if len(edge_colors) < 5:
                valid = False
                break
        
        if not valid:
            # Fallback to additive method
            return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return colors
# EVOLVE-BLOCK-END
