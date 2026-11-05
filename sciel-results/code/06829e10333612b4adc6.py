import numpy as np
import random
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    # We will use a decomposition based on splitting the 5x5 matrices into 4x4 and 1x1 blocks.
    # The 4x4 part will use Strassen's algorithm (rank 28).
    # The boundary parts will use schoolbook (rank 45).
    # Total rank = 28 + 45 = 73.
    
    rank = 73
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1

    # 1. 4x4 Strassen Decomposition (Rank 28)
    # Indices for 4x4: i,j in 0..3
    # Blocks:
    # A11: i,j in 0..1
    # A12: i in 0..1, j in 2..3
    # A21: i in 2..3, j in 0..1
    # A22: i,j in 2..3
    
    # We need to expand the 7 matrix multiplications into scalar multiplications.
    # Each 2x2 matmul has rank 4.
    # So 7 * 4 = 28 terms.
    
    # Strassen formulas:
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
    
    # We will generate the terms for each Mk.
    # For a 2x2 matmul C = AB, the terms are:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # We need to map the block indices to global indices.
    # Block (bi, bj) corresponds to i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
    
    def get_global_idx(bi, bj, li, lj):
        return (2*bi + li) * 5 + (2*bj + lj)
        
    # We will iterate through the 7 Strassen steps.
    # For each step, we have a product of two 2x2 matrices (which are linear combinations of A and B blocks).
    # We need to expand (Sum A) * (Sum B) into scalar terms.
    # This is (Sum a_i A_i) * (Sum b_j B_j) = Sum a_i b_j (A_i B_j).
    # Each A_i B_j is a 2x2 matmul, which expands to 4 scalar terms.
    
    # This is getting complex to hardcode manually.
    # I will use a simpler approach: just use schoolbook for the 4x4 part too, but that's rank 16.
    # Wait, 4x4 schoolbook is 16.
    # Strassen is 28.
    # So schoolbook is better for 4x4!
    # Rank 16 < 28.
    
    # So let's use schoolbook for the 4x4 part.
    # Rank 16 for 4x4.
    # Boundary rank 45.
    # Total rank 61.
    
    # Let's restart with Schoolbook for 4x4.
    
    rank = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # 1. 4x4 Schoolbook (Rank 16)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 5 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # 2. Boundary Terms
    # C_i5 = sum_{k=1}^4 A_ik B_k5 + A_i5 B_55
    # C_5j = sum_{k=1}^4 A_5k B_kj + A_55 B_5j
    # C_55 = sum_{k=1}^4 A_5k B_k5 + A_55 B_55
    
    # Terms for C_i5 (i=0..3)
    for i in range(4):
        # sum_{k=0}^3 A_ik B_k5
        for k in range(4):
            u_idx = i * 5 + 4 # j=4 is index 4
            v_idx = 4 * 5 + k # k is col index in B, row 4
            w_idx = k * 5 + i # w[k*n+i]
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
        # A_i5 B_55
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    # Terms for C_5j (j=0..3)
    for j in range(4):
        # sum_{k=0}^3 A_5k B_kj
        for k in range(4):
            u_idx = 4 * 5 + k # i=4, j=k
            v_idx = k * 5 + j # row k, col j
            w_idx = j * 5 + 4 # w[j*n+4]
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
        # A_55 B_5j
        u_idx = 4 * 5 + 4
        v_idx = 4 * 5 + j
        w_idx = j * 5 + 4
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
        
    # Terms for C_55
    # sum_{k=0}^3 A_5k B_k5
    for k in range(4):
        u_idx = 4 * 5 + k
        v_idx = 4 * 5 + k
        w_idx = 4 * 5 + 4
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    # A_55 B_55
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Check col
    if col != rank:
        # Adjust if needed
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
