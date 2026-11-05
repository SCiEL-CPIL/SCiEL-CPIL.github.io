import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    total_rank = 85
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Strassen 2x2x2 factors
    S_U = [[1,0,1,0,1,-1,0],
           [0,0,0,0,1,0,1],
           [0,1,0,0,0,1,0],
           [1,1,0,1,0,0,-1]]
    S_V = [[1,1,0,-1,0,1,0],
           [0,0,1,0,0,1,0],
           [0,0,0,1,0,0,1],
           [1,0,-1,0,1,0,1]]
    S_W = [[1,0,0,1,-1,0,1],
           [0,0,1,0,1,0,0],
           [0,1,0,1,0,0,0],
           [1,-1,1,0,0,1,0]]
    
    # Recursive block multiplication
    # mul_block(a_i0, a_i1, b_j0, b_j1, c_k0, c_k1)
    # Computes C[a_i0:a_i1, c_k0:c_k1] += A[a_i0:a_i1, b_j0:b_j1] * B[b_j0:b_j1, c_k0:c_k1]
    
    def mul_block(a_i0, a_i1, b_j0, b_j1, c_k0, c_k1):
        nonlocal col
        ai = a_i1 - a_i0
        bj = b_j1 - b_j0
        ck = c_k1 - c_k0
        
        if ai == 1 and bj == 1 and ck == 1:
            U[a_i0 * 5 + b_j0, col] = 1
            V[b_j0 * 5 + c_k0, col] = 1
            W[c_k0 * 4 + a_i0, col] = 1
            col += 1
            return
        
        if ai == 2 and bj == 2 and ck == 2:
            # Strassen
            for r in range(7):
                for lr in range(2):
                    for lc in range(2):
                        val = S_U[lr * 2 + lc][r]
                        if val:
                            U[(a_i0+lr) * 5 + (b_j0+lc), col+r] += val
                for lr in range(2):
                    for lc in range(2):
                        val = S_V[lr * 2 + lc][r]
                        if val:
                            V[(b_j0+lr) * 5 + (c_k0+lc), col+r] += val
                for lr in range(2):
                    for lc in range(2):
                        val = S_W[lr * 2 + lc][r]
                        if val:
                            W[(c_k0+lc) * 4 + (a_i0+lr), col+r] += val
            col += 7
            return
        
        # General case: schoolbook
        for i in range(ai):
            for j in range(bj):
                for k in range(ck):
                    U[(a_i0+i) * 5 + (b_j0+j), col] = 1
                    V[(b_j0+j) * 5 + (c_k0+k), col] = 1
                    W[(c_k0+k) * 4 + (a_i0+i), col] = 1
                    col += 1
    
    # 4x4x4 core: use 2+2 partition recursively
    # Split into 2x2x2 blocks
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                mul_block(2*bi, 2*bi+2, 2*bj, 2*bj+2, 2*bk, 2*bk+2)
    # This gives 8 * 7 = 56, not 49!
    
    # The issue: nested Strassen requires the 4x4x4 to be decomposed as
    # 7 Strassen terms, each of which is a 2x2x2 Strassen.
    # This is different from 8 independent 2x2x2 blocks.
