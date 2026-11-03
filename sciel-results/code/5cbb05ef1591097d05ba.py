import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_base():
    """
    Returns U, V, W for 2x2x2 matrix multiplication (Rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1 -> W[11], W[22]
    # Note: W index is k*2+i.
    # C11: k=0, i=0 -> W[0]
    # C22: k=1, i=1 -> W[3]
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C21 += M2 -> k=0, i=1 -> W[1]
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12 += M3 -> k=1, i=0 -> W[2]?
    # C12: i=0, k=1. W index k*2+i = 1*2+0 = 2.
    # Wait, my W index mapping above:
    # C11 (i=0,k=0) -> W[0]
    # C12 (i=0,k=1) -> W[2]
    # C21 (i=1,k=0) -> W[1]
    # C22 (i=1,k=1) -> W[3]
    
    # Let's re-verify W indices for M3.
    # C12 gets M3. i=0, k=1. Index 1*2+0 = 2.
    W[2, 2] = 1
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    # C11: i=0, k=0 -> W[0]
    # C21: i=1, k=0 -> W[1]
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C12 += M5, C22 -= M5
    # C12: i=0, k=1 -> W[2]
    # C22: i=1, k=1 -> W[3]
    W[2, 4] = 1
    W[3, 4] = -1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    # C22: i=1, k=1 -> W[3]
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    # C11: i=0, k=0 -> W[0]
    W[0, 6] = 1
    
    return U, V, W

def _strassen_recurse(n):
    if n == 2:
        return _strassen_base()
    
    # Split n into two halves
    h = n // 2
    U_sub, V_sub, W_sub = _strassen_recurse(h)
    
    # U_sub is (h*h) x 7
    # We need to construct U (n*n) x 49
    # The 7 macro terms correspond to the 7 columns of U_sub.
    # Each macro term involves linear combinations of the 4 blocks of A.
    
    # Strassen combinations for A blocks:
    # A = [[A00, A01], [A10, A11]]
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    # We need to map the local indices of U_sub to global indices.
    # U_sub[i_local, j_local, c_local]
    # Global index I, J.
    # I = 2*i_block + i_local
    # J = 2*j_block + j_local
    
    # For each macro term c_macro (0..6):
    #   For each local term c_local (0..6):
    #       c_global = c_macro * 7 + c_local
    #       The values in U are determined by the block coefficients and U_sub.
    
    U = np.zeros((n*n, 49), dtype=int)
    V = np.zeros((n*n, 49), dtype=int)
    W = np.zeros((n*n, 49), dtype=int)
    
    # Define block mappings for A (U), B (V), C (W)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # Coefficients for each macro term c:
    # c=0: A00+1, A11+1
    # c=1: A10+1, A11+1
    # c=2: A00+1
    # c=3: A11+1
    # c=4: A00+1, A01+1
    # c=5: A10+1, A00-1
    # c=6: A01+1, A11-1
    
    A_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    # B blocks:
    # c=0: B00+1, B11+1
    # c=1: B00+1
    # c=2: B01+1, B11-1
    # c=3: B10+1, B00-1
    # c=4: B11+1
    # c=5: B00+1, B01+1
    # c=6: B10+1, B11+1
    
    B_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # C blocks (W):
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_sub[k_local, i_local, c_local]
    # Global W index: k_block, i_block.
    # K = 2*k_block + k_local
    # I = 2*i_block + i_local
    
    # For each macro term c_macro:
    #   The result is added to specific C blocks with coefficients.
    #   c=0 (M1): C00+1, C11+1
    #   c=1 (M2): C10+1, C11-1
    #   c=2 (M3): C01+1, C11+1
    #   c=3 (M4): C00+1, C10+1
    #   c=4 (M5): C01-1, C11-1  (Wait, C01=M3+M5, C11=M1-M2+M3+M6. M5 is in C01? No.)
    #   Let's re-check Strassen output formulas:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    
    C_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): -1}, # M2
        {(0,1): 1, (1,1): 1},  # M3
        {(0,0): 1, (1,0): 1},  # M4
        {(0,0): -1, (0,1): 1}, # M5 (C00 -= M5, C01 += M5)
        {(1,1): 1},            # M6 (C11 += M6)
        {(0,0): 1}             # M7 (C00 += M7)
    ]
    
    for c_macro in range(7):
        for c_local in range(7):
            c_global = c_macro * 7 + c_local
            
            # Fill U
            for (ib, jb), coeff in A_coeffs[c_macro].items():
                # ib, jb are block indices (0 or 1)
                # U_sub has indices i_local, j_local
                for i_local in range(h):
                    for j_local in range(h):
                        val = U_sub[i_local, j_local, c_local] * coeff
                        if val != 0:
                            I = ib * h + i_local
                            J = jb * h + j_local
                            U[I, J, c_global] += val # Wait, U is 2D in code, but logically 3D.
                            # In the final return, U is (n*n, rank).
                            # So U[I*n+J, c_global] += val
                            U[I*n + J, c_global] += val
                            
            # Fill V
            for (ib, jb), coeff in B_coeffs[c_macro].items():
                for i_local in range(h):
                    for j_local in range(h):
                        val = V_sub[i_local, j_local, c_local] * coeff
                        if val != 0:
                            I = ib * h + i_local
                            J = jb * h + j_local
                            V[I*n + J, c_global] += val

            # Fill W
            for (ib, jb), coeff in C_coeffs[c_macro].items():
                # ib is k_block, jb is i_block
                for i_local in range(h):
                    for j_local in range(h):
                        # W_sub index: k_local, i_local
                        val = W_sub[i_local, j_local, c_local] * coeff # Note: W_sub[k, i]
                        # In _strassen_base, W[k*2+i, c]
                        # So W_sub row index is k_local*h + i_local?
                        # No, h=2 for base.
                        # W_sub is h*h x 7.
                        # Row index in W_sub is k_local*h + i_local.
                        val = W_sub[i_local*h + j_local, c_local] * coeff
                        if val != 0:
                            K = ib * h + i_local
                            I = jb * h + j_local
                            W[K*n + I, c_global] += val

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Get 4x4x4 decomposition
    U4, V4, W4 = _strassen_recurse(4)
    # U4: 16x49, V4: 16x49, W4: 16x49
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((28, rank), dtype=int)
    W = np.zeros((28, rank), dtype=int)
    
    # First 49 columns: k=0..3
    U[:, 0:49] = U4
    # V4 rows 0..15 correspond to j*4+k, k=0..3
    V[0:16, 0:49] = V4
    # W4 rows 0..15 correspond to k*4+i, k=0..3
    W[0:16, 0:49] = W4
    
    # Next 49 columns: k=4..7
    U[:, 49:98] = U4
    # V4 needs to be shifted.
    # V4 has non-zeros at j*4+k (k=0..3).
    # We need non-zeros at j*4+(k+4).
    # This is equivalent to adding 4 to the row index.
    V_shifted = np.zeros((28, 49), dtype=int)
    V_shifted[4:20, :] = V4 # Rows 4..19
    V[49:98, :] = V_shifted # Wait, V is 28x98.
    # V[:, 49:98] = V_shifted
    
    # W4 needs to be shifted.
    # W4 has non-zeros at k*4+i (k=0..3).
    # We need non-zeros at (k+4)*4+i = k*4+i+16.
    W_shifted = np.zeros((28, 49), dtype=int)
    W_shifted[16:28, :] = W4[0:12, :] # Rows 16..27 (12 rows)
    # W4 has 16 rows. We only need the first 12 (k=0,1,2 -> k_global=4,5,6).
    # k=3 -> k_global=7 (out of bounds).
    W[:, 49:98] = W_shifted
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
