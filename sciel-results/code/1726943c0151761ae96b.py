"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11=1, A22=1
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11=1, B22=1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11+=1, C22+=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21 += M2
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12 += M3
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11+=M4, C21+=M4
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1 # C12+=M5, C11-=M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds 4x4x4 decomposition using Strassen. Rank 49.
    """
    n, m, p = 4, 4, 4
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((n*m, 49))
    V = np.zeros((m*p, 49))
    W = np.zeros((p*n, 49))
    
    # Strassen steps for 4x4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map the 2x2 indices to 4x4 indices.
    # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # A11 is rows 0-1, cols 0-1.
    # A12 is rows 0-1, cols 2-3.
    # etc.
    
    # For each Strassen step c (0..6), we have a 2x2 multiplication.
    # The U factor for the 2x2 multiplication is U2.
    # The actual U for 4x4 is formed by combining the blocks of A.
    
    # Let's define the linear combinations for X_c (from A) and Y_c (from B).
    # X_c is a 2x2 matrix.
    # Y_c is a 2x2 matrix.
    
    # We will iterate over c (0..6) and d (0..6).
    # For each pair, we add a rank-1 term.
    
    # Coefficients for X_c (A blocks)
    # X1 = A11 + A22
    # X2 = A21 + A22
    # X3 = A11
    # X4 = A22
    # X5 = A11 + A12
    # X6 = A21 - A11
    # X7 = A12 - A22
    
    # Coefficients for Y_c (B blocks)
    # Y1 = B11 + B22
    # Y2 = B11
    # Y3 = B12 - B22
    # Y4 = B21 - B11
    # Y5 = B22
    # Y6 = B11 + B12
    # Y7 = B21 + B22
    
    # We represent X_c as a 4x4 matrix with zeros in non-participating blocks?
    # No, U2 has shape (4, 7) where rows are i*2+j for the 2x2 matrix.
    # We need to map the 2x2 indices (i,j) in U2 to 4x4 indices (I,i', J,j').
    
    # Let's do this explicitly.
    # For c in 0..6:
    #   For d in 0..6:
    #     col = c*7 + d
    #     # U part
    #     # U2[d] is the d-th column of U2.
    #     # U2[i*2+j, d] gives the coefficient for A_{ij} in the 2x2 matrix X_c.
    #     # But X_c is formed from A blocks.
    #     # So U[i*4+j, col] = sum over blocks?
    
    # Actually, it's easier to construct the 4x4 U matrix by placing the 2x2 U2 into the appropriate blocks.
    
    # Let's define the block structure.
    # A = [[A11, A12], [A21, A22]]
    # A11 is at [0:2, 0:2]
    # A12 is at [0:2, 2:4]
    # A21 is at [2:4, 0:2]
    # A22 is at [2:4, 2:4]
    
    # For X1 = A11 + A22:
    # The 2x2 matrix X1 has entries X1[i,j] = A11[i,j] + A22[i,j].
    # So in the 4x4 U matrix, the entries for rows corresponding to A11 and A22 are set.
    # Specifically, for a given d (Strassen factor index), U2[i*2+j, d] is the coefficient for the (i,j) entry of the 2x2 matrix.
    # So for X1, the coefficient for A11[i,j] is U2[i*2+j, d].
    # And for A22[i,j] is U2[i*2+j, d].
    
    # So U[0*4 + 0, col] += U2[0, d] # A11[0,0]
    # U[0*4 + 1, col] += U2[1, d] # A11[0,1]
    # U[1*4 + 0, col] += U2[2, d] # A11[1,0]
    # U[1*4 + 1, col] += U2[3, d] # A11[1,1]
    # U[2*4 + 2, col] += U2[0, d] # A22[0,0] (global row 2, col 2)
    # ...
    
    # This is getting tedious to hardcode. Let's write a helper.
    
    def add_block_U(U, block_r, block_c, u2_col, col):
        # block_r, block_c in {0,1}
        # u2_col is the column in U2 (0..3)
        # We add u2_col's values to the corresponding 2x2 block in U
        for i in range(2):
            for j in range(2):
                global_i = block_r * 2 + i
                global_j = block_c * 2 + j
                U[global_i * 4 + global_j, col] += U2[u2_col, d] # Wait, u2_col is i*2+j?
                # U2 is (4, 7). Rows are i*2+j.
                # So U2[i*2+j, d] is the value.
                
    # Let's refine.
    # U2 row index = i*2+j.
    
    # Define X_c blocks
    # c=0 (M1): X=A11+A22. Blocks: (0,0) with +1, (1,1) with +1.
    # c=1 (M2): X=A21+A22. Blocks: (1,0) with +1, (1,1) with +1.
    # c=2 (M3): X=A11. Blocks: (0,0) with +1.
    # c=3 (M4): X=A22. Blocks: (1,1) with +1.
    # c=4 (M5): X=A11+A12. Blocks: (0,0) with +1, (0,1) with +1.
    # c=5 (M6): X=A21-A11. Blocks: (1,0) with +1, (0,0) with -1.
    # c=6 (M7): X=A12-A22. Blocks: (0,1) with +1, (1,1) with -1.
    
    X_blocks = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    
    Y_blocks = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    
    W_coeff = [
        [(0,0,1), (1,1,1)], # M1 -> C11, C22
        [(1,0,1)],          # M2 -> C21
        [(0,1,1)],          # M3 -> C12
        [(0,0,1), (1,0,1)], # M4 -> C11, C21
        [(0,1,1), (0,0,-1)],# M5 -> C12, C11(-)
        [(1,1,1)],          # M6 -> C22
        [(0,0,1)]           # M7 -> C11
    ]
    
    for c in range(7):
        for d in range(7):
            col = c * 7 + d
            # U
            for br, bc, coeff in X_blocks[c]:
                for i in range(2):
                    for j in range(2):
                        gi = br * 2 + i
                        gj = bc * 2 + j
                        U[gi * 4 + gj, col] += coeff * U2[i * 2 + j, d]
                        
            # V
            for br, bc, coeff in Y_blocks[c]:
                for i in range(2):
                    for j in range(2):
                        gi = br * 2 + i
                        gj = bc * 2 + j
                        V[gi * 4 + gj, col] += coeff * V2[i * 2 + j, d]
                        
            # W
            for br, bc, coeff in W_coeff[c]:
                for i in range(2):
                    for j in range(2):
                        gi = br * 2 + i
                        gj = bc * 2 + j
                        W[gi * 4 + gj, col] += coeff * W2[i * 2 + j, d]
                        
    return U, V, W

def _build_4x4x2_schoolbook():
    """
    Builds 4x4x2 schoolbook decomposition. Rank 32.
    """
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Get 4x4x4 Strassen decomposition
    U1, V1, W1 = _build_4x4x4_strassen()
    rank1 = 49
    
    # Get 4x4x2 Schoolbook decomposition
    U2, V2, W2 = _build_4x4x2_schoolbook()
    rank2 = 32
    
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill U, V
    U[:, :rank1] = U1
    U[:, rank1:] = U2
    
    V[:, :rank1] = V1
    V[:, rank1:] = V2
    
    # Fill W
    # W1 corresponds to k=0..3
    # W2 corresponds to k=0..1 (local), which maps to k=4..5 (global)
    # W1 shape (4*4, 49) -> rows k*4+i for k=0..3
    # W2 shape (2*4, 32) -> rows k*4+i for k=0..1
    # We need to place W2 in W[4*4:6*4, rank1:]
    
    W[:4*4, :rank1] = W1
    
    # Map W2 to global indices
    for k_local in range(2):
        k_global = 4 + k_local
        W[k_global*4 : (k_global+1)*4, rank1:] = W2[k_local*4 : (k_local+1)*4, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

