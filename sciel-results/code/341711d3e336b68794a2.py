"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x10.
    Strategy: Use Strassen on 2x2 blocks of the middle dimension (5 blocks of size 2),
    plus schoolbook for the remaining middle dimension entry.
    Rank = 5*7 + 1*20 = 35 + 20 = 55.
    """
    n, m, p = 2, 3, 10
    
    # We'll try to find a decomposition with rank < 55 using a search approach.
    # For now, return the known rank-55 decomposition.
    
    rank = 55
    U_mat = np.zeros((n * m, rank), dtype=float)  # 6 x 55
    V_mat = np.zeros((m * p, rank), dtype=float)  # 30 x 55
    W_mat = np.zeros((p * n, rank), dtype=float)  # 20 x 55
    
    idx = 0
    
    # Term 1: Strassen on 5 blocks of 2x2 (middle dimension cols 0,1; 2,3; 4,5; 6,7; 8,9)
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices (2x3 matrix, using cols 0,1 for this block)
        a00 = 0   # A[0,0]
        a01 = 1   # A[0,1]
        a10 = 3   # A[1,0]
        a11 = 4   # A[1,1]
        
        # B indices (3x10 matrix, using rows 0,1 for this block)
        b0k0 = k0      # B[0,k0]
        b0k1 = k1      # B[0,k1]
        b1k0 = 10 + k0 # B[1,k0]
        b1k1 = 10 + k1 # B[1,k1]
        
        # W indices (output C is 2x10, index = k*2 + i)
        c0k0 = 2 * k0
        c0k1 = 2 * k1
        c1k0 = 2 * k0 + 1
        c1k1 = 2 * k1 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00, idx] += 1
        U_mat[a11, idx] += 1
        V_mat[b0k0, idx] += 1
        V_mat[b1k1, idx] += 1
        W_mat[c0k0, idx] += 1
        W_mat[c1k1, idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10, idx] += 1
        U_mat[a11, idx] += 1
        V_mat[b0k0, idx] += 1
        W_mat[c1k0, idx] += 1
        W_mat[c1k1, idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00, idx] += 1
        V_mat[b0k1, idx] += 1
        V_mat[b1k1, idx] -= 1
        W_mat[c0k1, idx] += 1
        W_mat[c1k1, idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11, idx] += 1
        V_mat[b1k0, idx] += 1
        V_mat[b0k0, idx] -= 1
        W_mat[c0k0, idx] += 1
        W_mat[c1k0, idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00, idx] += 1
        U_mat[a01, idx] += 1
        V_mat[b1k1, idx] += 1
        W_mat[c0k0, idx] -= 1
        W_mat[c0k1, idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10, idx] += 1
        U_mat[a00, idx] -= 1
        V_mat[b0k0, idx] += 1
        V_mat[b0k1, idx] += 1
        W_mat[c1k1, idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01, idx] += 1
        U_mat[a11, idx] -= 1
        V_mat[b1k0, idx] += 1
        V_mat[b1k1, idx] += 1
        W_mat[c0k0, idx] += 1
        idx += 1
    
    # Term 2: Schoolbook for j=2 (middle dimension row 2)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 20 + k
            w_idx = 2 * k + i
            
            U_mat[u_idx, idx] += 1
            V_mat[v_idx, idx] += 1
            W_mat[w_idx, idx] += 1
            idx += 1
            
    # Verify the decomposition is correct
    # The tensor T[i,j,k] = 1 if i,j,k are valid indices for matrix mult
    # We need: sum_r U[i,r] * V[j,r] * W[k,r] = delta_{i,j,k} for the multiplication tensor
    
    # Convert to lists for return
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {'U': U_list, 'V': V_list, 'W': W_list}
# EVOLVE-BLOCK-END
