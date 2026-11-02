"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a block-structured approach to reduce rank below schoolbook 32.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    We decompose the multiplication C = A @ B where:
    - A is 2x4, B is 4x4, C is 2x4
    - C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strategy: Split the 4x4 matrix B into two 2x4 blocks (top and bottom rows),
    and A into two 2x2 blocks (left and right columns).
    
    C = A_left @ B_top + A_right @ B_bottom
    
    Each 2x2 @ 2x4 multiplication can be done with rank 14 (two Strassen 2x2x2).
    Total rank = 14 + 14 = 28.
    
    Actually, let me think more carefully.
    
    A is 2x4: A = [A_L | A_R] where A_L, A_R are 2x2
    B is 4x4: B = [B_T; B_B] where B_T, B_B are 2x4
    
    C = A_L @ B_T + A_R @ B_B
    
    A_L @ B_T: 2x2 @ 2x4 = 2x4 result
    A_R @ B_B: 2x2 @ 2x4 = 2x4 result
    
    For 2x2 @ 2x4:
    Let X be 2x2, Y be 2x4, Z = X @ Y (2x4)
    Z[:, 0:2] = X @ Y[:, 0:2] (2x2 @ 2x2)
    Z[:, 2:4] = X @ Y[:, 2:4] (2x2 @ 2x2)
    
    Each 2x2 @ 2x2 can be done with Strassen (rank 7).
    So 2x2 @ 2x4 needs rank 14.
    
    Total: 14 + 14 = 28.
    
    Let me implement the Strassen decomposition for 2x2 @ 2x2.
    
    For X @ Y where X, Y, Z are 2x2:
    X = [[x00, x01], [x10, x11]]
    Y = [[y00, y01], [y10, y11]]
    Z = [[z00, z01], [z10, z11]]
    
    Strassen:
    M1 = (x00 + x11)(y00 + y11)
    M2 = (x10 + x11)y00
    M3 = x00(y01 - y11)
    M4 = x11(y10 - y00)
    M5 = (x00 + x01)y11
    M6 = (x10 - x00)(y00 + y01)
    M7 = (x01 - x11)(y10 + y11)
    
    z00 = M1 + M4 - M5 + M7
    z01 = M3 + M5
    z10 = M2 + M4
    z11 = M1 - M2 + M3 + M6
    
    In tensor decomposition terms, for each multiplication M_t, we have:
    U factors: linear combinations of X elements
    V factors: linear combinations of Y elements
    W factors: contribute to Z elements
    
    Let me build the full rank-28 decomposition.
    
    Indexing:
    A is 2x4, flattened as A[i*m + j] where i in {0,1}, j in {0,1,2,3}
    So A[0]=A[0,0], A[1]=A[0,1], A[2]=A[0,2], A[3]=A[0,3]
       A[4]=A[1,0], A[5]=A[1,1], A[6]=A[1,2], A[7]=A[1,3]
    
    B is 4x4, flattened as B[j*p + k] where j,k in {0,1,2,3}
    So B[0]=B[0,0], B[1]=B[0,1], B[2]=B[0,2], B[3]=B[0,3]
       B[4]=B[1,0], B[5]=B[1,1], B[6]=B[1,2], B[7]=B[1,3]
       B[8]=B[2,0], B[9]=B[2,1], B[10]=B[2,2], B[11]=B[2,3]
       B[12]=B[3,0], B[13]=B[3,1], B[14]=B[3,2], B[15]=B[3,3]
    
    C is 2x4, but with trace convention W[k*n + i] where n=2
    So W[0]=C[0,0], W[1]=C[0,1], W[2]=C[0,2], W[3]=C[0,3]
       W[4]=C[1,0], W[5]=C[1,1], W[6]=C[1,2], W[7]=C[1,3]
    
    Wait, the trace convention is W[k*n + i]. Let me re-read.
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W index = k*n + i where k is the column of C and i is the row of C.
    C has dimensions n x p = 2 x 4.
    So W[k*2 + i] for k in {0,1,2,3}, i in {0,1}.
    
    W[0] = C[0,0] (k=0, i=0)
    W[1] = C[1,0] (k=0, i=1)
    W[2] = C[0,1] (k=1, i=0)
    W[3] = C[1,1] (k=1, i=1)
    W[4] = C[0,2] (k=2, i=0)
    W[5] = C[1,2] (k=2, i=1)
    W[6] = C[0,3] (k=3, i=0)
    W[7] = C[1,3] (k=3, i=1)
    
    OK so W is indexed by (k, i) -> k*2 + i.
    
    Now let me implement the rank-28 decomposition.
    
    Block 1: A_L @ B_T where A_L = A[:, 0:2] (2x2), B_T = B[0:2, :] (2x4)
    Block 2: A_R @ B_B where A_R = A[:, 2:4] (2x2), B_B = B[2:4, :] (2x4)
    
    For Block 1:
    A_L = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
           = [[A[0], A[1]], [A[4], A[5]]]
    
    B_T = [[B[0,0], B[0,1], B[0,2], B[0,3]],
           [B[1,0], B[1,1], B[1,2], B[1,3]]]
           = [[B[0], B[1], B[2], B[3]],
              [B[4], B[5], B[6], B[7]]]
    
    C_L = A_L @ B_T (2x4)
    C_L[:, 0:2] = A_L @ B_T[:, 0:2] (2x2 @ 2x2)
    C_L[:, 2:4] = A_L @ B_T[:, 2:4] (2x2 @ 2x2)
    
    Similarly for Block 2.
    
    Let me define the Strassen decomposition for a 2x2 @ 2x2 multiplication.
    
    For X @ Y = Z where X, Y, Z are 2x2:
    X indices: x00, x01, x10, x11
    Y indices: y00, y01, y10, y11
    Z indices: z00, z01, z10, z11
    
    The 7 Strassen products:
    M1 = (x00 + x11) * (y00 + y11)
    M2 = (x10 + x11) * y00
    M3 = x00 * (y01 - y11)
    M4 = x11 * (y10 - y00)
    M5 = (x00 + x01) * y11
    M6 = (x10 - x00) * (y00 + y01)
    M7 = (x01 - x11) * (y10 + y11)
    
    Z contributions:
    z00 += M1 + M4 - M5 + M7
    z01 += M3 + M5
    z10 += M2 + M4
    z11 += M1 - M2 + M3 + M6
    
    In tensor form, each M_t gives one rank-1 term.
    
    Let me build this systematically.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    col = 0
    
    # Helper: add a Strassen 2x2@2x2 decomposition
    # X is a 2x2 submatrix of A, Y is a 2x2 submatrix of B, Z is a 2x2 submatrix of C
    # x_indices: list of 4 U indices for X[0,0], X[0,1], X[1,0], X[1,1]
    # y_indices: list of 4 V indices for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
    # z_indices: list of 4 W indices for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
    def add_strassen(x_idx, y_idx, z_idx, start_col):
        nonlocal col
        # M1 = (x00 + x11) * (y00 + y11)
        U[x_idx[0]][start_col] += 1
        U[x_idx[3]][start_col] += 1
        V[y_idx[0]][start_col] += 1
        V[y_idx[3]][start_col] += 1
        W[z_idx[0]][start_col] += 1   # z00
        W[z_idx[3]][start_col] += 1   # z11
        col += 1
        
        # M2 = (x10 + x11) * y00
        U[x_idx[2]][start_col] += 1
        U[x_idx[3]][start_col] += 1
        V[y_idx[0]][start_col] += 1
        W[z_idx[2]][start_col] += 1   # z10
        W[z_idx[3]][start_col] -= 1   # z11
        col += 1
        
        # M3 = x00 * (y01 - y11)
        U[x_idx[0]][start_col] += 1
        V[y_idx[1]][start_col] += 1
        V[y_idx[3]][start_col] -= 1
        W[z_idx[1]][start_col] += 1   # z01
        W[z_idx[3]][start_col] += 1   # z11
        col += 1
        
        # M4 = x11 * (y10 - y00)
        U[x_idx[3]][start_col] += 1
        V[y_idx[2]][start_col] += 1
        V[y_idx[0]][start_col] -= 1
        W[z_idx[0]][start_col] += 1   # z00
        W[z_idx[2]][start_col] += 1   # z10
        col += 1
        
        # M5 = (x00 + x01) * y11
        U[x_idx[0]][start_col] += 1
        U[x_idx[1]][start_col] += 1
        V[y_idx[3]][start_col] += 1
        W[z_idx[0]][start_col] -= 1   # z00
        W[z_idx[1]][start_col] += 1   # z01
        col += 1
        
        # M6 = (x10 - x00) * (y00 + y01)
        U[x_idx[2]][start_col] += 1
        U[x_idx[0]][start_col] -= 1
        V[y_idx[0]][start_col] += 1
        V[y_idx[1]][start_col] += 1
        W[z_idx[3]][start_col] += 1   # z11
        col += 1
        
        # M7 = (x01 - x11) * (y10 + y11)
        U[x_idx[1]][start_col] += 1
        U[x_idx[3]][start_col] -= 1
        V[y_idx[2]][start_col] += 1
        V[y_idx[3]][start_col] += 1
        W[z_idx[0]][start_col] += 1   # z00
        col += 1
        
        return col
    
    # Block 1: A_L @ B_T
    # A_L = A[:, 0:2], indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    x_idx_L = [0, 1, 4, 5]
    
    # B_T = B[0:2, :], split into B_T[:, 0:2] and B_T[:, 2:4]
    # B_T[:, 0:2]: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    y_idx_T0 = [0, 1, 4, 5]
    # C_L[:, 0:2]: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    # W index: k*n+i, so C[i,k] -> W[k*2+i]
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    z_idx_L0 = [0, 2, 1, 3]
    
    # B_T[:, 2:4]: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    y_idx_T1 = [2, 3, 6, 7]
    # C_L[:, 2:4]: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    # C[0,2] -> W[2*2+0] = W[4]
    # C[0,3] -> W[3*2+0] = W[6]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    z_idx_L1 = [4, 6, 5, 7]
    
    col = add_strassen(x_idx_L, y_idx_T0, z_idx_L0, col)
    col = add_strassen(x_idx_L, y_idx_T1, z_idx_L1, col)
    
    # Block 2: A_R @ B_B
    # A_R = A[:, 2:4], indices: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    x_idx_R = [2, 3, 6, 7]
    
    # B_B = B[2:4, :], split into B_B[:, 0:2] and B_B[:, 2:4]
    # B_B[:, 0:2]: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    y_idx_B0 = [8, 9, 12, 13]
    # C_R[:, 0:2]: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    z_idx_R0 = [0, 2, 1, 3]
    
    # B_B[:, 2:4]: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    y_idx_B1 = [10, 11, 14, 15]
    # C_R[:, 2:4]: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    z_idx_R1 = [4, 6, 5, 7]
    
    col = add_strassen(x_idx_R, y_idx_B0, z_idx_R0, col)
    col = add_strassen(x_idx_R, y_idx_B1, z_idx_R1, col)
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
