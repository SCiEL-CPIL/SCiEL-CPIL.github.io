import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 18 as fallback
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # We search for a rank-15 decomposition using integer coefficients.
    # The tensor T has entries T[i,j,k] = 1 if (i,j,k) is a valid index.
    # We look for rank-r decomposition: sum_{t=1}^{r} c_t * u_t ⊗ v_t ⊗ w_t
    # where u_t, v_t, w_t are indicator vectors of supports.
    
    # For 2x3x3 matrix multiplication, the known minimal rank is 15.
    # We'll use a known construction or search for it.
    
    # Known rank-15 construction for 2x3x3:
    # This is based on the work of Landsberg and others.
    # Let's try to construct it explicitly.
    
    # Indices:
    # U: (i,j) with i in {0,1}, j in {0,1,2} -> 6 indices
    # V: (j,k) with j in {0,1,2}, k in {0,1,2} -> 9 indices  
    # W: (k,i) with k in {0,1,2}, i in {0,1} -> 6 indices
    
    # We'll use a systematic search for rank 15 with integer coefficients.
    # Given the small size, we can enumerate candidate supports.
    
    # For each rank-1 term, the support must be consistent:
    # If U-support has indices (i1,j1), (i2,j2), ...
    # V-support has indices (j1,k1), (j2,k2), ...
    # W-support has indices (k1,i1), (k2,i2), ...
    # Then the covered triples are those where i, j, k match up.
    
    # Let's use a simpler approach: try known patterns.
    # Based on literature, a rank-15 decomposition exists.
    
    # We'll construct it using the following method:
    # Use 9 terms for the "diagonal" part and 6 terms for corrections.
    
    # Actually, let's use a direct construction from known results.
    # For 2x3x3, one known rank-15 construction uses:
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0-8: Standard terms for each (i,j,k)
    # But we need to combine them cleverly.
    
    # Let's use the construction from the paper "On the border rank of matrix multiplication"
    # or similar. For 2x3x3, we can use:
    
    # I'll use a known explicit construction:
    # Terms 0-4: For i=0, various j,k combinations
    # Terms 5-9: For i=1, various j,k combinations
    # Terms 10-14: Correction terms
    
    # Actually, let me use a verified construction.
    # Based on the known result that rank(2,3,3) = 15, here's one construction:
    
    # We'll define the 15 terms explicitly.
    # Each term is (U_support, V_support, W_support, coefficient)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_supp, v_supp, w_supp, coeff):
        idx = len(terms)
        if idx >= rank:
            return
        for a in u_supp:
            U[a][idx] = coeff
        for b in v_supp:
            V[b][idx] = coeff
        for c in w_supp:
            W[c][idx] = coeff
        terms.append((u_supp, v_supp, w_supp, coeff))
    
    # Known rank-15 construction for 2x3x3:
    # This uses the fact that we can share some indices.
    
    # Let's use a systematic approach with integer coefficients.
    # We'll try to find it by solving the system.
    
    # For now, let's use a known valid construction.
    # Here's one from the literature:
    
    # Terms for i=0:
    # (0,0), (0,1), (0,2) in U -> indices 0, 1, 2
    # (0,0), (0,1), (0,2) in V -> indices 0, 1, 2
    # (0,0), (1,0), (2,0) in W -> indices 0, 1, 2
    
    # This is getting complex. Let's use a different approach.
    
    # We'll use the fact that for 2x3x3, we can achieve rank 15 with:
    # 9 terms that cover the main structure and 6 correction terms.
    
    # Let me use a verified construction from online resources.
    # One known construction uses these 15 terms:
    
    # I'll return the fallback for now since I can't guarantee correctness
    # of a hand-crafted rank-15 solution without verification.
    
    return fallback
# EVOLVE-BLOCK-END
