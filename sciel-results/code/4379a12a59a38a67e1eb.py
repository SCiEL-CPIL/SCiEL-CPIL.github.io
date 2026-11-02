import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _get_strassen_222_terms():
    """
    Returns list of (U, V, W) vectors for 2x2x2 Strassen decomposition.
    Local indices: A:0-3, B:0-3, C:0-3
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A01-A11)(B10+B11) -> C00 += 1
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _get_schoolbook_142_terms():
    """
    Returns list of (U, V, W) vectors for 1x4x2 schoolbook decomposition.
    Local indices: A:0-3, B:0-7, C:0-1
    B is 4x2: B00,B01,B10,B11,B20,B21,B30,B31 -> 0,1,2,3,4,5,6,7
    C is 1x2: C0, C1 -> 0, 1
    """
    terms = []
    
    # C0 = A0*B00 + A1*B10 + A2*B20 + A3*B30
    terms.append(([1,0,0,0], [1,0,0,0,0,0,0,0], [1,0]))
    terms.append(([0,1,0,0], [0,0,1,0,0,0,0,0], [1,0]))
    terms.append(([0,0,1,0], [0,0,0,0,1,0,0,0], [1,0]))
    terms.append(([0,0,0,1], [0,0,0,0,0,0,1,0], [1,0]))
    
    # C1 = A0*B01 + A1*B11 + A2*B21 + A3*B31
    terms.append(([1,0,0,0], [0,1,0,0,0,0,0,0], [0,1]))
    terms.append(([0,1,0,0], [0,0,0,1,0,0,0,0], [0,1]))
    terms.append(([0,0,1,0], [0,0,0,0,0,1,0,0], [0,1]))
    terms.append(([0,0,0,1], [0,0,0,0,0,0,0,1], [0,1]))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18
    
    terms = []
    
    def add_term(u_local, v_local, w_local, u_map, v_map, w_map):
        u_global = [0] * U_dim
        v_global = [0] * V_dim
        w_global = [0] * W_dim
        
        for i, val in enumerate(u_local):
            if val != 0:
                u_global[u_map[i]] = val
                
        for i, val in enumerate(v_local):
            if val != 0:
                v_global[v_map[i]] = val
                
        for i, val in enumerate(w_local):
            if val != 0:
                w_global[w_map[i]] = val
                
        terms.append((u_global, v_global, w_global))

    strassen_terms = _get_strassen_222_terms()
    schoolbook_terms = _get_schoolbook_142_terms()
    
    # Block 1: A1 (rows 0,1) * B1 (cols 0,1) -> C11 (rows 0,1; cols 0,1)
    # A1 indices: 0,1,4,5
    u_map_1 = [0, 1, 4, 5]
    # B1 indices: 0,1, 6,7, 12,13, 18,19
    v_map_1 = [0, 1, 6, 7, 12, 13, 18, 19]
    # C11 indices in W (k*3+i):
    # k=0,1; i=0,1
    # k=0,i=0 -> 0; k=0,i=1 -> 1
    # k=1,i=0 -> 3; k=1,i=1 -> 4
    w_map_1 = [0, 1, 3, 4]
    
    for u, v, w in strassen_terms:
        add_term(u, v, w, u_map_1, v_map_1, w_map_1)
        
    # Block 2: A1 * B2 (cols 2,3) -> C12 (rows 0,1; cols 2,3)
    u_map_2 = [0, 1, 4, 5]
    # B2 indices: 2,3, 8,9, 14,15, 20,21
    v_map_2 = [2, 3, 8, 9, 14, 15, 20, 21]
    # C12 indices in W:
    # k=2,3; i=0,1
    # k=2,i=0 -> 6; k=2,i=1 -> 7
    # k=3,i=0 -> 9; k=3,i=1 -> 10
    w_map_2 = [6, 7, 9, 10]
    
    for u, v, w in strassen_terms:
        add_term(u, v, w, u_map_2, v_map_2, w_map_2)
        
    # Block 3: A1 * B3 (cols 4,5) -> C13 (rows 0,1; cols 4,5)
    u_map_3 = [0, 1, 4, 5]
    # B3 indices: 4,5, 10,11, 16,17, 22,23
    v_map_3 = [4, 5, 10, 11, 16, 17, 22, 23]
    # C13 indices in W:
    # k=4,5; i=0,1
    # k=4,i=0 -> 12; k=4,i=1 -> 13
    # k=5,i=0 -> 15; k=5,i=1 -> 16
    w_map_3 = [12, 13, 15, 16]
    
    for u, v, w in strassen_terms:
        add_term(u, v, w, u_map_3, v_map_3, w_map_3)
        
    # Block 4: A2 (row 2) * B1 (cols 0,1) -> C21 (row 2; cols 0,1)
    u_map_4 = [8, 9, 10, 11]
    v_map_4 = [0, 1, 6, 7, 12, 13, 18, 19]
    # C21 indices in W:
    # k=0,1; i=2
    # k=0,i=2 -> 2; k=1,i=2 -> 5
    w_map_4 = [2, 5]
    
    for u, v, w in schoolbook_terms:
        add_term(u, v, w, u_map_4, v_map_4, w_map_4)
        
    # Block 5: A2 * B2 (cols 2,3) -> C22 (row 2; cols 2,3)
    u_map_5 = [8, 9, 10, 11]
    v_map_5 = [2, 3, 8, 9, 14, 15, 20, 21]
    # C22 indices in W:
    # k=2,3; i=2
    # k=2,i=2 -> 8; k=3,i=2 -> 11
    w_map_5 = [8, 11]
    
    for u, v, w in schoolbook_terms:
        add_term(u, v, w, u_map_5, v_map_5, w_map_5)
        
    # Block 6: A2 * B3 (cols 4,5) -> C23 (row 2; cols 4,5)
    u_map_6 = [8, 9, 10, 11]
    v_map_6 = [4, 5, 10, 11, 16, 17, 22, 23]
    # C23 indices in W:
    # k=4,5; i=2
    # k=4,i=2 -> 14; k=5,i=2 -> 17
    w_map_6 = [14, 17]
    
    for u, v, w in schoolbook_terms:
        add_term(u, v, w, u_map_6, v_map_6, w_map_6)
        
    rank = len(terms)
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for r, (u, v, w) in enumerate(terms):
        for a in range(U_dim):
            if u[a] != 0:
                U[a][r] = u[a]
        for b in range(V_dim):
            if v[b] != 0:
                V[b][r] = v[b]
        for c in range(W_dim):
            if w[c] != 0:
                W[c][r] = w[c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
