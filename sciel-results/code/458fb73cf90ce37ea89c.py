"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    # Schoolbook rank = n*m*p = 72
    # We'll use a Strassen-inspired decomposition for the top-left 2x2 block of A
    # combined with schoolbook for the rest.
    
    # A is n x m = 3x3
    # B is m x p = 3x8
    # C is n x p = 3x8
    
    # Split A into:
    # A = [[A11, A12],     where A11 is 2x2, A12 is 2x1
    #      [A21, A22]]     where A21 is 1x2, A22 is 1x1
    
    # Split B into:
    # B = [[B1],           where B1 is 2x8
    #      [B2]]           where B2 is 1x8
    
    # C = A11*B1 + A12*B2 + A21*B1 + A22*B2
    
    # For A11*B1 (2x2 * 2x8), use Strassen with 4 blocks of 2x2 each:
    # B1 = [B11, B12, B13, B14] where each Bi is 2x2
    # For each 2x2*2x2 multiplication, Strassen uses 7 multiplications
    # Total for A11*B1: 4 * 7 = 28 multiplications
    
    # A12*B2: 2x1 * 1x8 = 2x8, schoolbook = 2*8 = 16 multiplications
    # A21*B1: 1x2 * 2x8 = 1x8, schoolbook = 2*8 = 16 multiplications
    # A22*B2: 1x1 * 1x8 = 1x8, schoolbook = 8 multiplications
    
    # Total rank = 28 + 16 + 16 + 8 = 68
    
    rank = 68
    
    # Initialize factor matrices
    U = np.zeros((n*m, rank))  # 9 x 68
    V = np.zeros((m*p, rank))  # 24 x 68
    W = np.zeros((p*n, rank))  # 24 x 68
    
    col = 0
    
    # --- Part 1: A11 * B1 using Strassen ---
    # A11 = [[A[0,0], A[0,1]],
    #        [A[1,0], A[1,1]]]
    # A indices: A[i*m + j] where i=0..1, j=0..1 -> indices 0,1,3,4
    # B1 = B[0:2, 0:8] -> B indices: row 0 (B[0:8]), row 1 (B[8:16])
    # C from A11*B1 contributes to C[0:2, 0:8]
    
    # B1 split into 4 blocks of 2x2:
    # B11 = B[0:2, 0:2] -> B indices 0,1,8,9
    # B12 = B[0:2, 2:4] -> B indices 2,3,10,11
    # B13 = B[0:2, 4:6] -> B indices 4,5,12,13
    # B14 = B[0:2, 6:8] -> B indices 6,7,14,15
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each block t in 0..3:
    for t in range(4):
        # B block t: rows 0,1; cols 2*t, 2*t+1
        # B indices for row 0: 2*t, 2*t+1
        # B indices for row 1: 8 + 2*t, 8 + 2*t + 1
        b00 = 2*t      # B[0, 2*t]
        b01 = 2*t + 1  # B[0, 2*t+1]
        b10 = 8 + 2*t      # B[1, 2*t]
        b11 = 8 + 2*t + 1  # B[1, 2*t+1]
        
        # A indices for A11:
        a00 = 0  # A[0,0]
        a01 = 1  # A[0,1]
        a10 = 3  # A[1,0]
        a11 = 4  # A[1,1]
        
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        V[b11, col] = 1
        # C contribution: C[0,2*t] += M1, C[0,2*t+1] -= M1, C[1,2*t] -= M1, C[1,2*t+1] += M1
        # W index: C[i*n + j] where i is row of C (0 or 1), j is col of C (2*t or 2*t+1)
        # But wait, W shape is (p*n, rank) = (24, 68). W[k*n + i, r]
        # The convention is W[k*n+i, r] where k is col of C (0..7), i is row of C (0..2)
        # So W index for C[i, k] is k*n + i = k*3 + i
        
        w_00 = (2*t) * 3 + 0      # C[0, 2*t]
        w_01 = (2*t+1) * 3 + 0    # C[0, 2*t+1]
        w_10 = (2*t) * 3 + 1      # C[1, 2*t]
        w_11 = (2*t+1) * 3 + 1    # C[1, 2*t+1]
        
        W[w_00, col] = 1
        W[w_01, col] = -1
        W[w_10, col] = -1
        W[w_11, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        # C contribution: C[1,2*t] += M2
        W[w_10, col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00, col] = 1
        V[b01, col] = 1
        V[b11, col] = -1
        # C contribution: C[0,2*t+1] += M3
        W[w_01, col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11, col] = 1
        V[b10, col] = 1
        V[b00, col] = -1
        # C contribution: C[0,2*t] += M4, C[1,2*t] += M4
        W[w_00, col] = 1
        W[w_10, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00, col] = 1
        U[a01, col] = 1
        V[b11, col] = 1
        # C contribution: C[0,2*t] -= M5, C[1,2*t+1] += M5
        W[w_00, col] = -1
        W[w_11, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10, col] = 1
        U[a00, col] = -1
        V[b00, col] = 1
        V[b01, col] = 1
        # C contribution: C[1,2*t+1] += M6
        W[w_11, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] = 1
        U[a11, col] = -1
        V[b10, col] = 1
        V[b11, col] = 1
        # C contribution: C[0,2*t] += M7
        W[w_00, col] = 1
        col += 1
    
    # --- Part 2: A12 * B2 (schoolbook) ---
    # A12 = [[A[0,2]],   -> A indices: 2, 5
    #        [A[1,2]]]
    # B2 = B[2, 0:8] -> B indices: 16, 17, 18, 19, 20, 21, 22, 23
    
    for i in range(2):  # rows 0, 1 of A12
        a_idx = i * 3 + 2  # A[i, 2]
        for k in range(8):
            b_idx = 16 + k  # B[2, k]
            # C[i, k] += A[i,2] * B[2,k]
            w_idx = k * 3 + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    # --- Part 3: A21 * B1 (schoolbook) ---
    # A21 = [[A[2,0], A[2,1]]] -> A indices: 6, 7
    # B1 = B[0:2, 0:8] -> B indices: 0..15
    
    for j in range(2):  # cols 0, 1 of A21
        a_idx = 6 + j  # A[2, j]
        for i in range(2):  # rows 0, 1 of B1
            for k in range(8):
                b_idx = i * 8 + k  # B[i, k]
                # C[2, k] += A[2,j] * B[i,k]
                w_idx = k * 3 + 2
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # --- Part 4: A22 * B2 (schoolbook) ---
    # A22 = A[2,2] -> A index: 8
    # B2 = B[2, 0:8] -> B indices: 16..23
    
    for k in range(8):
        b_idx = 16 + k  # B[2, k]
        w_idx = k * 3 + 2  # C[2, k]
        U[8, col] = 1
        V[b_idx, col] = 1
        W[w_idx, col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
