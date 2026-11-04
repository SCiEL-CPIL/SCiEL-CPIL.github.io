import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partitions for I, J, K
    # I: 3 -> [2, 1]
    # J: 5 -> [2, 2, 1]
    # K: 6 -> [2, 2, 2]
    
    I_parts = [[0, 2], [2, 3]]  # Start, End indices (0-indexed, end exclusive)
    J_parts = [[0, 2], [2, 4], [4, 5]]
    K_parts = [[0, 2], [2, 4], [4, 6]]
    
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    # Helper to add a block
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal U, V, W
        
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        # Determine rank for this block
        if ni == 2 and nj == 2 and nk == 2:
            # Strassen decomposition
            # Local indices: i in 0..1, j in 0..1, k in 0..1
            # Global indices: I = i_start + i, J = j_start + j, K = k_start + k
            
            # Strassen 7 terms for 2x2x2
            # U_local[i, j, r], V_local[j, k, r], W_local[k, i, r]
            # We need to map these to global U, V, W columns.
            
            # Strassen formulas (for C = AB):
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B22)
            # M4 = A22(B12 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We are decomposing the tensor T_ijk = 1.
            # The "variables" are the entries of A and B.
            # But here we are constructing the decomposition of the multiplication tensor itself.
            # The tensor T_ijk represents the coefficient of A_ij B_jk in C_ik.
            # For 2x2x2, T_ijk = 1 for all i,j,k.
            # We want to express this tensor as sum of 7 rank-1 tensors.
            
            # The standard Strassen decomposition of the 2x2x2 multiplication tensor:
            # U, V, W are 4x7 matrices (flattened i,j / j,k / k,i).
            
            # Let's define the 7 terms explicitly.
            # Term 1: (A11+A22)(B11+B22) -> C11, C22
            # Term 2: (A21+A22)B11 -> C21
            # Term 3: A11(B21-B22) -> C12, C22
            # Term 4: A22(B12-B11) -> C11, C21
            # Term 5: (A11+A12)B22 -> C11, C12
            # Term 6: (A21-A11)(B11+B12) -> C22
            # Term 7: (A12-A22)(B21+B22) -> C11
            
            # We need U[i,j], V[j,k], W[k,i] such that sum U V W = 1.
            
            # This is a known set of values.
            # See "Strassen's Algorithm" tensor decomposition.
            
            # Local U (2x2x7), V (2x2x7), W (2x2x7)
            # Indices: i,j in 0,1. k,i in 0,1.
            
            # Term 1:
            # U1: A11, A22 -> U[0,0]=1, U[1,1]=1, others 0.
            # V1: B11, B22 -> V[0,0]=1, V[1,1]=1, others 0.
            # W1: C11, C22 -> W[0,0]=1, W[1,1]=1, others 0.
            
            # Term 2:
            # U2: A21, A22 -> U[1,0]=1, U[1,1]=1.
            # V2: B11 -> V[0,0]=1.
            # W2: C21 -> W[1,0]=1.
            
            # Term 3:
            # U3: A11 -> U[0,0]=1.
            # V3: B21, B22 -> V[1,0]=1, V[1,1]=-1.
            # W3: C12, C22 -> W[0,1]=1, W[1,1]=1.
            
            # Term 4:
            # U4: A22 -> U[1,1]=1.
            # V4: B12, B11 -> V[0,1]=1, V[0,0]=-1.
            # W4: C11, C21 -> W[0,0]=1, W[1,0]=1.
            
            # Term 5:
            # U5: A11, A12 -> U[0,0]=1, U[0,1]=1.
            # V5: B22 -> V[1,1]=1.
            # W5: C11, C12 -> W[0,0]=1, W[0,1]=1.
            
            # Term 6:
            # U6: A21, A11 -> U[1,0]=1, U[0,0]=-1.
            # V6: B11, B12 -> V[0,0]=1, V[0,1]=1.
            # W6: C22 -> W[1,1]=1.
            
            # Term 7:
            # U7: A12, A22 -> U[0,1]=1, U[1,1]=-1.
            # V7: B21, B22 -> V[1,0]=1, V[1,1]=1.
            # W7: C11 -> W[0,0]=1.

            # We add these 7 terms to the global matrices.
            for r in range(7):
                U_col = np.zeros(ni * nj)
                V_col = np.zeros(nj * nk)
                W_col = np.zeros(nk * ni)
                
                if r == 0:
                    U_col[0*2+0] = 1; U_col[1*2+1] = 1
                    V_col[0*2+0] = 1; V_col[1*2+1] = 1
                    W_col[0*2+0] = 1; W_col[1*2+1] = 1
                elif r == 1:
                    U_col[1*2+0] = 1; U_col[1*2+1] = 1
                    V_col[0*2+0] = 1
                    W_col[1*2+0] = 1
                elif r == 2:
                    U_col[0*2+0] = 1
                    V_col[1*2+0] = 1; V_col[1*2+1] = -1
                    W_col[0*2+1] = 1; W_col[1*2+1] = 1
                elif r == 3:
                    U_col[1*2+1] = 1
                    V_col[0*2+1] = 1; V_col[0*2+0] = -1
                    W_col[0*2+0] = 1; W_col[1*2+0] = 1
                elif r == 4:
                    U_col[0*2+0] = 1; U_col[0*2+1] = 1
                    V_col[1*2+1] = 1
                    W_col[0*2+0] = 1; W_col[0*2+1] = 1
                elif r == 5:
                    U_col[1*2+0] = 1; U_col[0*2+0] = -1
                    V_col[0*2+0] = 1; V_col[0*2+1] = 1
                    W_col[1*2+1] = 1
                elif r == 6:
                    U_col[0*2+1] = 1; U_col[1*2+1] = -1
                    V_col[1*2+0] = 1; V_col[1*2+1] = 1
                    W_col[0*2+0] = 1

                # Append to global matrices
                U = np.column_stack((U, U_col))
                V = np.column_stack((V, V_col))
                W = np.column_stack((W, W_col))

        else:
            # Trivial decomposition for blocks with dim 1
            # Rank is 1 if any dim is 1?
            # If ni=1, nj=1, nk=1: Rank 1.
            # If ni=2, nj=1, nk=2: Rank 1.
            # We can just use 1 term.
            # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
            # This works because the product is 1*1*1 = 1.
            
            U_col = np.ones(ni * nj)
            V_col = np.ones(nj * nk)
            W_col = np.ones(nk * ni)
            
            U = np.column_stack((U, U_col))
            V = np.column_stack((V, V_col))
            W = np.column_stack((W, W_col))

    # Iterate through all blocks
    for i_idx, (i_start, i_end) in enumerate(I_parts):
        for j_idx, (j_start, j_end) in enumerate(J_parts):
            for k_idx, (k_start, k_end) in enumerate(K_parts):
                add_block(i_start, i_end, j_start, j_end, k_start, k_end)
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
