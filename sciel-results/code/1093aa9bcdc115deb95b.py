import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the U, V, W matrices by summing contributions from blocks.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] using Strassen (Rank 7)
    # Block 2: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2] using Naive (Rank 2*3*2=12)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:6] -> C[0:2, 2:6] using Naive (Rank 2*2*4=16)
    # Block 4: A[0:2, 2:5] * B[2:5, 2:6] -> C[0:2, 2:6] using Naive (Rank 2*3*4=24)
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2] using Naive (Rank 1*2*2=4)
    # Block 6: A[2:3, 2:5] * B[2:5, 0:2] -> C[2:3, 0:2] using Naive (Rank 1*3*2=6)
    # Block 7: A[2:3, 0:2] * B[0:2, 2:6] -> C[2:3, 2:6] using Naive (Rank 1*2*4=8)
    # Block 8: A[2:3, 2:5] * B[2:5, 2:6] -> C[2:3, 2:6] using Naive (Rank 1*3*4=12)
    
    total_rank = 7 + 12 + 16 + 24 + 4 + 6 + 8 + 12
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a naive block
    def add_naive_block(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Helper to add Strassen 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # Global indices:
    # A: i in [i0, i0+1], j in [j0, j0+1]
    # B: j in [j0, j0+1], k in [k0, k0+1]
    # C: i in [i0, i0+1], k in [k0, k0+1]
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A11: i0, j0; A22: i0+1, j0+1
        # B11: j0, k0; B22: j0+1, k0+1
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We define the 7 multiplications
        # 1. (A11+A22) * (B11+B22)
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        # Note: W for M1 contributes to C11 and C22
        # C11 += M1, C22 += M1
        # We need to distribute the W vector.
        # The decomposition is sum of rank-1 tensors.
        # For Strassen, we have 7 rank-1 tensors.
        # T1 = (e1+e2)_A x (e1+e2)_B x (e1_C x e1_C? No)
        # Actually, the output is linear combinations of the M's.
        # So we add 7 columns to U, V, W.
        # The W matrix will have the coefficients for the output.
        
        # Let's define the 7 terms explicitly.
        # Term 1: M1
        # U: A11, A22
        # V: B11, B22
        # W: C11 (coeff 1), C22 (coeff 1)
        # But W is a vector of size p*n.
        # W[k*n + i]
        # C11 is i=i0, k=k0.
        # C22 is i=i0+1, k=k0+1.
        # So W for M1 has 1 at k0*n+i0 and 1 at (k0+1)*n+(i0+1).
        
        # We will create a local U, V, W for the block and add to global.
        # But since we are adding to global col, we just set the values.
        
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A21+A22) * B11
        # A21: i0+1, j0; A22: i0+1, j0+1
        # B11: j0, k0
        # C21 += M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M3 = A11 * (B12-B22)
        # A11: i0, j0
        # B12: j0, k0+1; B22: j0+1, k0+1
        # C12 += M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M4 = A22 * (B21-B11)
        # A22: i0+1, j0+1
        # B21: j0+1, k0; B11: j0, k0
        # C11 += M4, C21 += M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5 = (A11+A12) * B22
        # A11: i0, j0; A12: i0, j0+1
        # B22: j0+1, k0+1
        # C11 -= M5, C12 += M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += -1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6 = (A21-A11) * (B11+B12)
        # A21: i0+1, j0; A11: i0, j0
        # B11: j0, k0; B12: j0, k0+1
        # C22 += M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7 = (A12-A22) * (B21+B22)
        # A12: i0, j0+1; A22: i0+1, j0+1
        # B21: j0+1, k0; B22: j0+1, k0+1
        # C11 += M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Apply Strassen to Block 1: i=0,1; j=0,1; k=0,1
    add_strassen_block(0, 0, 0)
    
    # Apply Naive to other blocks
    # Block 2: A[0:2, 2:5] * B[2:5, 0:2]
    add_naive_block(0, 2, 2, 5, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:6]
    add_naive_block(0, 2, 0, 2, 2, 6)
    
    # Block 4: A[0:2, 2:5] * B[2:5, 2:6]
    add_naive_block(0, 2, 2, 5, 2, 6)
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2]
    add_naive_block(2, 3, 0, 2, 0, 2)
    
    # Block 6: A[2:3, 2:5] * B[2:5, 0:2]
    add_naive_block(2, 3, 2, 5, 0, 2)
    
    # Block 7: A[2:3, 0:2] * B[0:2, 2:6]
    add_naive_block(2, 3, 0, 2, 2, 6)
    
    # Block 8: A[2:3, 2:5] * B[2:5, 2:6]
    add_naive_block(2, 3, 2, 5, 2, 6)
    
    return {"U": U, "V": V, "W": W}
