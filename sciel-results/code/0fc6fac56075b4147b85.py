import numpy as np
import time
import sys

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 by 3x7 matrix multiplication.
    
    Uses Strassen's algorithm for 2x2x2 blocks where possible.
    A is 2x3, B is 3x7, C is 2x7.
    Output C[k*n + i, r] following trace convention.
    
    Strategy:
    - Split columns of A into j=0,1 (2x2 block) and j=2 (1x1 block)
    - For j=0,1: split rows of B into pairs (0,1), (2,3), (4,5) using Strassen (rank 7 each),
      and row 6 using schoolbook (rank 4)
    - For j=2: use schoolbook for all 7 rows (rank 14)
    
    Total rank = 3*7 + 4 + 14 = 39
    """
    n, m, p = 2, 3, 7
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs_u=None, signs_v=None, signs_w=None):
        nonlocal col
        if signs_u is None:
            signs_u = [1] * len(u_indices)
        if signs_v is None:
            signs_v = [1] * len(v_indices)
        if signs_w is None:
            signs_w = [1] * len(w_indices)
        
        for idx, sign in zip(u_indices, signs_u):
            U[idx][col] += sign
        for idx, sign in zip(v_indices, signs_v):
            V[idx][col] += sign
        for idx, sign in zip(w_indices, signs_w):
            W[idx][col] += sign
        col += 1
    
    def add_strassen_2x2x2(k_start, j_start=0):
        """
        Add Strassen decomposition for 2x2x2 block.
        A rows i in {0,1}, cols j in {j_start, j_start+1}
        B rows j in {j_start, j_start+1}, cols k in {k_start, k_start+1}
        C rows k in {k_start, k_start+1}, cols i in {0,1}
        
        Indexing:
        A[i*m + j]
        B[j*p + k]
        C[k*n + i]
        """
        # A indices
        a00 = 0 * m + j_start
        a01 = 0 * m + j_start + 1
        a10 = 1 * m + j_start
        a11 = 1 * m + j_start + 1
        
        # B indices
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        # C indices
        c00 = k_start * n + 0
        c01 = (k_start + 1) * n + 0
        c10 = k_start * n + 1
        c11 = (k_start + 1) * n + 1
        
        # Strassen's 7 products:
        # M1 = (A00 + A11)(B00 + B11)
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2 = (A10 + A11)(B00)
        add_term([a10, a11], [b00], [c10, c11], signs_w=[1, -1])
        
        # M3 = (A00)(B01 - B11)
        add_term([a00], [b01, b11], [c01, c11], signs_v=[1, -1])
        
        # M4 = (A11)(B10 - B00)
        add_term([a11], [b10, b00], [c00, c10], signs_v=[1, -1])
        
        # M5 = (A00 + A01)(B11)
        add_term([a00, a01], [b11], [c00, c01], signs_w=[-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term([a10, a00], [b00, b01], [c11], signs_u=[1, -1])
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term([a01, a11], [b10, b11], [c00], signs_u=[1, -1])
    
    # Add 3 Strassen blocks for k pairs (0,1), (2,3), (4,5) with j=0,1
    add_strassen_2x2x2(0, 0)
    add_strassen_2x2x2(2, 0)
    add_strassen_2x2x2(4, 0)
    
    # Schoolbook for k=6 with j=0,1 (2x2x1)
    # C[k*2 + i] = sum over j of A[i*3+j] * B[j*7+k]
    # For k=6: C[12+i] = A[0,0]*B[0,6] + A[0,1]*B[1,6] + A[1,0]*B[0,6] + A[1,1]*B[1,6]... 
    # Wait, that's not right. Let me re-index.
    # C[k*n + i] for k=6: C[12 + i]
    # C[12] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[13] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    
    a00 = 0 * m + 0  # A[0,0]
    a01 = 0 * m + 1  # A[0,1]
    a10 = 1 * m + 0  # A[1,0]
    a11 = 1 * m + 1  # A[1,1]
    b06 = 0 * p + 6  # B[0,6]
    b16 = 1 * p + 6  # B[1,6]
    c12 = 6 * n + 0  # C[6,0]
    c13 = 6 * n + 1  # C[6,1]
    
    # 4 schoolbook terms for 2x2x1
    add_term([a00], [b06], [c12])
    add_term([a01], [b16], [c12])
    add_term([a10], [b06], [c13])
    add_term([a11], [b16], [c13])
    
    # Schoolbook for j=2 with all k (2x1x7)
    # C[k*n + i] includes term A[i*3+2] * B[2*7+k]
    # For each k in 0..6, i in 0..1:
    a02 = 0 * m + 2  # A[0,2]
    a12 = 1 * m + 2  # A[1,2]
    
    for k in range(7):
        b2k = 2 * p + k  # B[2,k]
        c_k0 = k * n + 0  # C[k,0]
        c_k1 = k * n + 1  # C[k,1]
        
        add_term([a02], [b2k], [c_k0])
        add_term([a12], [b2k], [c_k1])
    
    return {"U": U, "V": V, "W": W}
