import numpy as np
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Initialize factor matrices
    # We don't know the final rank yet, so we collect terms
    terms = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        terms.append((u_vec, v_vec, w_vec))
        
    # Define local Strassen factors for 2x2x2
    # Local indices i, j, k in {0, 1}
    # u_vec, v_vec, w_vec are lists of (local_idx, val)
    def get_strassen_terms(i_start, j_start, k_start):
        # m1 = (a+d)(e+h)
        u1 = [((i_start+0, j_start+0), 1), ((i_start+1, j_start+1), 1)]
        v1 = [((j_start+0, k_start+0), 1), ((j_start+1, k_start+1), 1)]
        w1 = [((k_start+0, i_start+0), 1), ((k_start+1, i_start+1), 1)]
        add_term(u1, v1, w1)
        
        # m2 = (c+d)e
        u2 = [((i_start+1, j_start+0), 1), ((i_start+1, j_start+1), 1)]
        v2 = [((j_start+0, k_start+0), 1)]
        w2 = [((k_start+0, i_start+1), 1), ((k_start+1, i_start+1), -1)]
        add_term(u2, v2, w2)
        
        # m3 = a(f-h)
        u3 = [((i_start+0, j_start+0), 1)]
        v3 = [((j_start+0, k_start+1), 1), ((j_start+1, k_start+1), -1)]
        w3 = [((k_start+1, i_start+0), 1), ((k_start+1, i_start+1), 1)]
        add_term(u3, v3, w3)
        
        # m4 = d(g-f)
        u4 = [((i_start+1, j_start+1), 1)]
        v4 = [((j_start+1, k_start+0), 1), ((j_start+0, k_start+0), -1)]
        w4 = [((k_start+0, i_start+0), 1)]
        add_term(u4, v4, w4)
        
        # m5 = (a+b)h
        u5 = [((i_start+0, j_start+0), 1), ((i_start+0, j_start+1), 1)]
        v5 = [((j_start+1, k_start+1), 1)]
        w5 = [((k_start+0, i_start+0), -1), ((k_start+1, i_start+0), 1)]
        add_term(u5, v5, w5)
        
        # m6 = (c+d)g
        u6 = [((i_start+1, j_start+0), 1), ((i_start+1, j_start+1), 1)]
        v6 = [((j_start+1, k_start+0), 1)]
        w6 = [((k_start+0, i_start+1), 1), ((k_start+1, i_start+1), 1)]
        add_term(u6, v6, w6)
        
        # m7 = (a-b)(e+f)
        u7 = [((i_start+0, j_start+0), 1), ((i_start+0, j_start+1), -1)]
        v7 = [((j_start+0, k_start+0), 1), ((j_start+1, k_start+0), 1)]
        w7 = [((k_start+0, i_start+0), 1), ((k_start+1, i_start+1), -1)]
        add_term(u7, v7, w7)

    def get_2x1x2_terms(i_start, j_start, k_start):
        # j is single, so j_start is the index
        # A is 2x1, B is 1x2
        # Terms:
        # 1. a e -> C00
        u1 = [((i_start+0, j_start), 1)]
        v1 = [((j_start, k_start+0), 1)]
        w1 = [((k_start+0, i_start+0), 1)]
        add_term(u1, v1, w1)
        
        # 2. a f -> C01
        u2 = [((i_start+0, j_start), 1)]
        v2 = [((j_start, k_start+1), 1)]
        w2 = [((k_start+1, i_start+0), 1)]
        add_term(u2, v2, w2)
        
        # 3. b e -> C10
        u3 = [((i_start+1, j_start), 1)]
        v3 = [((j_start, k_start+0), 1)]
        w3 = [((k_start+0, i_start+1), 1)]
        add_term(u3, v3, w3)
        
        # 4. b f -> C11
        u4 = [((i_start+1, j_start), 1)]
        v4 = [((j_start, k_start+1), 1)]
        w4 = [((k_start+1, i_start+1), 1)]
        add_term(u4, v4, w4)

    # Partitions
    i_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    j_pairs = [(0, 1), (2, 3)]
    j_singles = [4]
    
    for i_start, i_end in i_pairs:
        for k_start, k_end in k_pairs:
            for j_start, j_end in j_pairs:
                get_strassen_terms(i_start, j_start, k_start)
            for j in j_singles:
                get_2x1x2_terms(i_start, j, k_start)

    # Construct final matrices
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (i, j), val in u_list:
            U[i * m + j][r] = val
        for (j, k), val in v_list:
            V[j * p + k][r] = val
        for (k, i), val in w_list:
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
