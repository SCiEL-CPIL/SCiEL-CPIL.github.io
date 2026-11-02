import time
import numpy as np
import random

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # We construct a rank-39 decomposition.
    # The tensor is T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # U shape: (n*m, rank) = (6, 39)
    # V shape: (m*p, rank) = (21, 39)
    # W shape: (p*n, rank) = (14, 39)
    
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    # u_dict: { (i,j): coeff }
    # v_dict: { (j,k): coeff }
    # w_dict: { (i,k): coeff } -> flattened to k*n + i
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for (i, j), coeff in u_dict.items():
            U[i * m + j, col] += coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, col] += coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, col] += coeff
        col += 1

    # Strassen terms for 2x2x2 block
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # a=A00, b=A01, c=A10, d=A11
    # e=B00, f=B01, g=B10, h=B11
    # C00, C01, C10, C11
    
    def strassen_block(i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets for i, j, k indices
        # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        
        # m1 = (a+d)(e+h) -> u: A00+A11, v: B00+B11, w: C00+C11
        u1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        w1 = {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
        add_term(u1, v1, w1)
        
        # m2 = (c+d)e -> u: A10+A11, v: B00, w: C10
        u2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        v2 = {(j_off, k_off): 1}
        w2 = {(i_off+1, k_off): 1}
        add_term(u2, v2, w2)
        
        # m3 = a(f-h) -> u: A00, v: B01-B11, w: C01
        u3 = {(i_off, j_off): 1}
        v3 = {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1}
        w3 = {(i_off, k_off+1): 1}
        add_term(u3, v3, w3)
        
        # m4 = d(g-e) -> u: A11, v: B10-B00, w: C10+C11
        u4 = {(i_off+1, j_off+1): 1}
        v4 = {(j_off+1, k_off): 1, (j_off, k_off): -1}
        w4 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): 1}
        add_term(u4, v4, w4)
        
        # m5 = (a+b)h -> u: A00+A01, v: B11, w: C00+C01
        u5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        v5 = {(j_off+1, k_off+1): 1}
        w5 = {(i_off, k_off): 1, (i_off, k_off+1): 1}
        add_term(u5, v5, w5)
        
        # m6 = (c-a)(e+f) -> u: A10-A00, v: B00+B01, w: C11
        u6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        v6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        w6 = {(i_off+1, k_off+1): 1}
        add_term(u6, v6, w6)
        
        # m7 = (b-d)(g+h) -> u: A01-A11, v: B10+B11, w: C00+C10
        u7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        v7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        w7 = {(i_off, k_off): 1, (i_off+1, k_off): 1}
        add_term(u7, v7, w7)

    # Schoolbook block for 2x2x1 (i in {0,1}, j in {0,1}, k in {0})
    def schoolbook_2x2x1(i_off, j_off, k_off):
        for i in range(2):
            for j in range(2):
                u = {(i_off+i, j_off+j): 1}
                v = {(j_off+j, k_off): 1}
                w = {(i_off+i, k_off): 1}
                add_term(u, v, w)

    # Schoolbook block for 2x1x2 (i in {0,1}, j in {0}, k in {0,1})
    def schoolbook_2x1x2(i_off, j_off, k_off):
        for i in range(2):
            for k in range(2):
                u = {(i_off+i, j_off): 1}
                v = {(j_off, k_off+k): 1}
                w = {(i_off+i, k_off+k): 1}
                add_term(u, v, w)

    # Schoolbook block for 2x1x5 (i in {0,1}, j in {0}, k in {0..4})
    def schoolbook_2x1x5(i_off, j_off, k_off):
        for i in range(2):
            for k in range(5):
                u = {(i_off+i, j_off): 1}
                v = {(j_off, k_off+k): 1}
                w = {(i_off+i, k_off+k): 1}
                add_term(u, v, w)

    # Apply blocks
    # 1. 2x2x2 at i=0, j=0, k=0
    strassen_block(0, 0, 0)
    
    # 2. 2x2x2 at i=0, j=0, k=2
    strassen_block(0, 0, 2)
    
    # 3. 2x2x2 at i=0, j=0, k=4
    strassen_block(0, 0, 4)
    
    # 4. 2x2x1 at i=0, j=0, k=6
    schoolbook_2x2x1(0, 0, 6)
    
    # 5. 2x1x2 at i=0, j=2, k=0
    schoolbook_2x1x2(0, 2, 0)
    
    # 6. 2x1x5 at i=0, j=2, k=2
    schoolbook_2x1x5(0, 2, 2)
    
    # Convert to lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
