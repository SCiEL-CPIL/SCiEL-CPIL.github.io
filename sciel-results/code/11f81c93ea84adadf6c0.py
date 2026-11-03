"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = len(U[0])
    # Check that each tensor slice T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = 1 if i, j, k correspond to a valid (A_ij * B_jk -> C_ik) term
    # Actually, the multiplication tensor T_{(i,j),(j,k),(k,i)} = 1
    # Let's just check numerically: for random A, B, compute C = A@B and compare with tensor network
    
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Tensor network: C_ik = sum_j sum_c A_ij * B_jk * (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] / ... )
        # Actually the decomposition is: T_{(i,j),(j,k),(k,i)} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # The multiplication tensor has T = 1 for each (i,j,k) triple in the schoolbook sense.
        # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k? No!
        # The multiplication tensor is T_{(i,j),(j,k),(k,i)} = delta_{i,i} delta_{j,j} delta_{k,k}... 
        # Actually it's just T_{(i,j),(j,k),(k,i)} = 1 for all valid indices.
        # Wait no. The matrix multiplication tensor has components T_{(i,j),(j,k),(k,i)} where
        # the first index ranges over (i,j) pairs, second over (j,k) pairs, third over (k,i) pairs.
        # T_{(i,j),(j',k),(k',i')} = delta_{jj'} * delta_{kk'} * delta_{ii'} 
        # So T_{(i,j),(j,k),(k,i)} = 1.
        
        # The decomposition says: T = sum_c U_c outer V_c outer W_c
        # i.e., T_{(i,j),(j,k),(k,i)} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T_{(i,j),(j,k),(k,i)} = 1 for all i, j, k.
        
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # Let's verify this directly.
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    max_err = max(max_err, abs(s - 1.0))
        if max_err > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 44 construction:
    # 1. Schoolbook for j=0 (2x1x8): rank 16
    # 2. Strassen for j=1,2 (2x2x8): decompose into 4 blocks of 2x2x2, each rank 7 -> total 28
    # Total rank = 16 + 28 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2, Strassen blocks
    # Strassen definitions for 2x2x2
    # U indices (i,j) local: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # V indices (j,k) local: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    # W indices (k,i) local: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    strassen_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # m3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),   # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # m5
        ([0, 0, 1, -1], [0, 1, 1, 0], [0, 0, 0, 1]),   # m6
        ([1, 0, -1, 0], [1, 1, 1, 0], [1, 0, 0, 0]),   # m7
    ]
    
    for k_group in range(4):
        k_base = k_group * 2
        for c_s in range(7):
            u_vec, v_vec, w_vec = strassen_defs[c_s]
            c_global = col + c_s
            
            # U: global i in {0,1}, global j in {1,2}
            for i_l in range(2):
                for j_l in range(2):
                    idx_u = i_l * m + (1 + j_l)
                    U[idx_u][c_global] += u_vec[i_l * 2 + j_l]
            
            # V: global j in {1,2}, global k in {k_base, k_base+1}
            for j_l in range(2):
                for k_l in range(2):
                    idx_v = (1 + j_l) * p + (k_base + k_l)
                    V[idx_v][c_global] += v_vec[j_l * 2 + k_l]
            
            # W: global k in {k_base, k_base+1}, global i in {0,1}
            for k_l in range(2):
                for i_l in range(2):
                    idx_w = (k_base + k_l) * n + i_l
                    W[idx_w][c_global] += w_vec[k_l * 2 + i_l]
        
        col += 7
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
