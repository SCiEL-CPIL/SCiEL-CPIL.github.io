"""Rewritten program using transposed problem M_{10,3,2} approach with block partition."""
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Original problem: multiply A (2x3) by B (3x10) to get C (2x10)
    # Transpose: multiply A^T (3x2) by B^T (10x3) to get C^T (10x2)
    # This is a 10 x 3 x 2 matrix multiplication problem
    
    # Using the approach: split middle dimension m=3 into 2+1
    # For the 2x2 part: use Strassen (rank 7 per 2x2 block)
    # For the 1x1 remainder: schoolbook (rank 20)
    
    # In the transposed problem:
    # A^T is 3x2, B^T is 10x3, C^T is 10x2
    # We split the middle dimension of B^T (which has 3 columns) into first 2 and last 1
    
    # Actually, let's stick with the original formulation but try to optimize
    # The current approach gives rank 55. Let's see if we can do better.
    
    # For 2x3 * 3x10:
    # - First 2 columns of A (2x2) with first 2 rows of B (2x10): this is 2x2 * 2x10
    #   We can decompose the 2x10 as 5 blocks of 2x2, each needing rank 7 (Strassen)
    #   Total: 5 * 7 = 35
    # - Last column of A (2x1) with last row of B (1x10): this is 2x1 * 1x10
    #   This needs rank 20 (each of 2*10=20 products)
    #   Total: 20
    # Grand total: 35 + 20 = 55
    
    # This is already optimal for this decomposition. Let's verify the implementation.
    
    n, m, p = 2, 3, 10
    rank = 55
    
    U_mat = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    idx = 0
    
    # Term 1: Strassen for the 2x2 * 2x10 part (5 blocks of 2x2)
    # A indices: i*3 + j for i in 0,1 and j in 0,1
    # B indices: j*10 + k for j in 0,1 and k in 0,9
    # C indices: k*2 + i for k in 0,9 and i in 0,1
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices
        a00 = 0 * 3 + 0  # 0
        a01 = 0 * 3 + 1  # 1
        a10 = 1 * 3 + 0  # 3
        a11 = 1 * 3 + 1  # 4
        
        # B indices
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # W indices
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
    
    # Term 2: Schoolbook for j=2 (last column of A, last row of B)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
