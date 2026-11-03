import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_off, j_off, k_off):
        # i_off, j_off, k_off are the starting indices for the 2x2x2 block
        
        # Strassen terms for 2x2x2
        # A is 2x2, B is 2x2, C is 2x2
        # C = AB
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will add 7 terms
        
        # Term 1: M1
        u1 = [0] * (n*m)
        v1 = [0] * (m*p)
        w1 = [0] * (p*n)
        
        u1[i_off*m + j_off] = 1
        u1[(i_off+1)*m + (j_off+1)] = 1
        
        v1[j_off*p + k_off] = 1
        v1[(j_off+1)*p + (k_off+1)] = 1
        
        w1[k_off*n + i_off] = 1
        w1[(k_off+1)*n + (i_off+1)] = 1
        
        add_term(u1, v1, w1)
        
        # Term 2: M2
        u2 = [0] * (n*m)
        v2 = [0] * (m*p)
        w2 = [0] * (p*n)
        
        u2[(i_off+1)*m + j_off] = 1
        u2[(i_off+1)*m + (j_off+1)] = 1
        
        v2[j_off*p + k_off] = 1
        
        w2[k_off*n + (i_off+1)] = 1
        
        add_term(u2, v2, w2)
        
        # Term 3: M3
        u3 = [0] * (n*m)
        v3 = [0] * (m*p)
        w3 = [0] * (p*n)
        
        u3[i_off*m + j_off] = 1
        
        v3[j_off*p + (k_off+1)] = 1
        v3[(j_off+1)*p + (k_off+1)] = -1
        
        w3[(k_off+1)*n + i_off] = 1
        
        add_term(u3, v3, w3)
        
        # Term 4: M4
        u4 = [0] * (n*m)
        v4 = [0] * (m*p)
        w4 = [0] * (p*n)
        
        u4[(i_off+1)*m + (j_off+1)] = 1
        
        v4[(j_off+1)*p + k_off] = 1
        v4[j_off*p + k_off] = -1
        
        w4[k_off*n + (i_off+1)] = 1
        
        add_term(u4, v4, w4)
        
        # Term 5: M5
        u5 = [0] * (n*m)
        v5 = [0] * (m*p)
        w5 = [0] * (p*n)
        
        u5[i_off*m + j_off] = 1
        u5[i_off*m + (j_off+1)] = 1
        
        v5[(j_off+1)*p + (k_off+1)] = 1
        
        w5[(k_off+1)*n + i_off] = 1
        
        add_term(u5, v5, w5)
        
        # Term 6: M6
        u6 = [0] * (n*m)
        v6 = [0] * (m*p)
        w6 = [0] * (p*n)
        
        u6[(i_off+1)*m + j_off] = 1
        u6[i_off*m + j_off] = -1
        
        v6[j_off*p + k_off] = 1
        v6[j_off*p + (k_off+1)] = 1
        
        w6[(k_off+1)*n + (i_off+1)] = 1
        
        add_term(u6, v6, w6)
        
        # Term 7: M7
        u7 = [0] * (n*m)
        v7 = [0] * (m*p)
        w7 = [0] * (p*n)
        
        u7[i_off*m + (j_off+1)] = 1
        u7[(i_off+1)*m + (j_off+1)] = -1
        
        v7[(j_off+1)*p + k_off] = 1
        v7[(j_off+1)*p + (k_off+1)] = 1
        
        w7[k_off*n + i_off] = 1
        
        add_term(u7, v7, w7)
        
    # Helper to add a 1x2x2 block (schoolbook)
    def add_1x2x2_block(i_fix, j_off, k_off):
        # i is fixed, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        
        # Terms:
        # A_{i, j_off} B_{j_off, k_off} C_{i, k_off}
        # A_{i, j_off+1} B_{j_off+1, k_off} C_{i, k_off}
        # A_{i, j_off} B_{j_off, k_off+1} C_{i, k_off+1}
        # A_{i, j_off+1} B_{j_off+1, k_off+1} C_{i, k_off+1}
        
        u1 = [0] * (n*m)
        v1 = [0] * (m*p)
        w1 = [0] * (p*n)
        u1[i_fix*m + j_off] = 1
        v1[j_off*p + k_off] = 1
        w1[k_off*n + i_fix] = 1
        add_term(u1, v1, w1)
        
        u2 = [0] * (n*m)
        v2 = [0] * (m*p)
        w2 = [0] * (p*n)
        u2[i_fix*m + (j_off+1)] = 1
        v2[(j_off+1)*p + k_off] = 1
        w2[k_off*n + i_fix] = 1
        add_term(u2, v2, w2)
        
        u3 = [0] * (n*m)
        v3 = [0] * (m*p)
        w3 = [0] * (p*n)
        u3[i_fix*m + j_off] = 1
        v3[j_off*p + (k_off+1)] = 1
        w3[(k_off+1)*n + i_fix] = 1
        add_term(u3, v3, w3)
        
        u4 = [0] * (n*m)
        v4 = [0] * (m*p)
        w4 = [0] * (p*n)
        u4[i_fix*m + (j_off+1)] = 1
        v4[(j_off+1)*p + (k_off+1)] = 1
        w4[(k_off+1)*n + i_fix] = 1
        add_term(u4, v4, w4)
        
    # Partition:
    # i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    add_strassen_block(0, 0, 0)
    
    # i in {0,1}, j in {2,3}, k in {0,1} -> Strassen
    add_strassen_block(0, 2, 0)
    
    # i in {0,1}, j in {0,1}, k in {2,3} -> Strassen
    add_strassen_block(0, 0, 2)
    
    # i in {0,1}, j in {2,3}, k in {2,3} -> Strassen
    add_strassen_block(0, 2, 2)
    
    # i in {2}, j in {0,1}, k in {0,1} -> 1x2x2
    add_1x2x2_block(2, 0, 0)
    
    # i in {2}, j in {2,3}, k in {0,1} -> 1x2x2
    add_1x2x2_block(2, 2, 0)
    
    # i in {2}, j in {0,1}, k in {2,3} -> 1x2x2
    add_1x2x2_block(2, 0, 2)
    
    # i in {2}, j in {2,3}, k in {2,3} -> 1x2x2
    add_1x2x2_block(2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
