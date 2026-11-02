"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: 2x1x6 (j=0)
    col = 0
    for i in range(2):
        for k in range(6):
            U[5*i + 0][col] = 1
            V[0*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1

    def add_strassen_block(j_offset, k_offset, col_start):
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] -= 1
        
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] -= 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 0][col_start] -= 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        col_start += 1
        U[5*0 + j_offset + 1][col_start] += 1
        U[5*1 + j_offset + 1][col_start] -= 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        
        return col_start + 1

    col = add_strassen_block(1, 0, 12)
    col = add_strassen_block(1, 2, col)
    col = add_strassen_block(1, 4, col)
    col = add_strassen_block(3, 0, col)
    col = add_strassen_block(3, 2, col)
    col = add_strassen_block(3, 4, col)
    
    # Try to reduce rank by finding linear dependencies
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We want to check if any column r is a linear combination of others
    # Form the matrix M where M[a,b] = sum_r U[i_a, r] * V[j_b, r] * W[k_c, r] for each (i,j,k)
    # Actually, let's flatten: for each rank-1 term r, we have a vector in R^(nm*mp*pn)
    # But that's too large. Instead, check linear dependence of the "slices"
    
    # Alternative: check if the U, V, W columns are linearly dependent in a way
    # that preserves the tensor
    # For now, try simple elimination: if two columns of U are proportional, 
    # and same for V and W, we can merge
    # This is complex, so let's just return the current result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
