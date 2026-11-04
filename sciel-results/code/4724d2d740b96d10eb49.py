import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    start_time = time.time()
    deadline = 140.0

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()

    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a00 + a11)(b00 + b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
        # Correction: W indices are k*2+i.
        # C00 is k=0, i=0 -> idx 0. C11 is k=1, i=1 -> idx 3.
        # So W[0,0]=1, W[3,0]=1 is correct for M1 contribution to C00 and C11?
        # Wait. C00 = M1 + M4 - M5 + M7. C11 = M1 - M2 + M3 + M6.
        # So M1 contributes +1 to C00 and +1 to C11.
        
        # M2 = (a10 + a11) b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 += M2 (k=1, i=0 -> idx 2)
        W[3, 1] = -1 # C11 -= M2 (k=1, i=1 -> idx 3)
        
        # M3 = a00 (b01 - b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 += M3 (k=0, i=1 -> idx 1)
        W[3, 2] = 1 # C11 += M3
        
        # M4 = a11 (b10 - b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1 # C00 += M4
        W[2, 3] = 1 # C10 += M4
        
        # M5 = (a00 + a01) b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C00 -= M5
        W[1, 4] = 1 # C01 += M5
        
        # M6 = (a10 - a00) (b00 + b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11 += M6
        
        # M7 = (a01 - a11) (b10 + b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00 += M7
        
        return U, V, W

    def get_2x1x2_rank4():
        # A (2x1), B (1x2) -> C (2x2)
        # U shape (2, 4), V shape (2, 4), W shape (4, 4)
        # U[i], V[j], W[k*2+i]
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # T1 = a0 b0 -> C00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # T2 = a0 b1 -> C01
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        # T3 = a1 b0 -> C10
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        # T4 = a1 b1 -> C11
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

    def get_2x3x1_rank6():
        # A (2x3), B (3x1) -> C (2x1)
        # U shape (6, 6), V shape (3, 6), W shape (2, 6)
        # U[i*3+j], V[j], W[i]
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    def get_2x3x3_rank17():
        # Combine 2x2x2 (rank 7), 2x1x2 (rank 4), 2x3x1 (rank 6)
        # Total rank 17
        U17 = np.zeros((6, 17))
        V17 = np.zeros((9, 17))
        W17 = np.zeros((6, 17))
        
        # Block 1: 2x2x2 (cols 0-1 of B, cols 0-1 of C)
        U_s, V_s, W_s = get_strassen_2x2x2()
        # Map U_s (4x7) to U17 (6x17)
        # U_s rows: 0,1,2,3 (i*2+j for i,j in 0,1)
        # U17 rows: 0..5 (i*3+j for i in 0,1, j in 0,1,2)
        # j=0,1 correspond to U17 cols 0,1 and 3,4? No.
        # U17 row index i*3+j.
        # For j=0: rows 0, 3.
        # For j=1: rows 1, 4.
        # U_s row 0 (i=0,j=0) -> U17 row 0
        # U_s row 1 (i=0,j=1) -> U17 row 1
        # U_s row 2 (i=1,j=0) -> U17 row 3
        # U_s row 3 (i=1,j=1) -> U17 row 4
        
        U17[0, 0:7] = U_s[0, :]
        U17[1, 0:7] = U_s[1, :]
        U17[3, 0:7] = U_s[2, :]
        U17[4, 0:7] = U_s[3, :]
        
        # Map V_s (4x7) to V17 (9x17)
        # V_s rows: 0,1,2,3 (j*2+k for j,k in 0,1)
        # V17 rows: 0..8 (j*3+k for j in 0,1,2, k in 0,1,2)
        # k=0,1 correspond to V17 cols 0,1 and 3,4 and 6,7?
        # V17 row index j*3+k.
        # For j=0: rows 0, 1, 2. We use k=0,1 -> rows 0, 1.
        # For j=1: rows 3, 4, 5. We use k=0,1 -> rows 3, 4.
        
        V17[0, 0:7] = V_s[0, :] # j=0, k=0
        V17[1, 0:7] = V_s[1, :] # j=0, k=1
        V17[3, 0:7] = V_s[2, :] # j=1, k=0
        V17[4, 0:7] = V_s[3, :] # j=1, k=1
        
        # Map W_s (4x7) to W17 (6x17)
        # W_s rows: 0,1,2,3 (k*2+i for k,i in 0,1)
        # W17 rows: 0..5 (k*2+i for k in 0,1,2, i in 0,1)
        # k=0,1 correspond to W17 rows 0,1 and 2,3.
        
        W17[0, 0:7] = W_s[0, :]
        W17[1, 0:7] = W_s[1, :]
        W17[2, 0:7] = W_s[2, :]
        W17[3, 0:7] = W_s[3, :]
        
        # Block 2: 2x1x2 (col 2 of B, cols 2 of C? No, 2x1x2 is A(2x1) B(1x2))
        # This handles the interaction of A(:,2) with B(2,0) and B(2,1)?
        # Wait, the decomposition of 2x3x3 into 2x2x2 + 2x1x2 + 2x3x1 is:
        # A = [A1 A2], B = [B1; B2] where A1 is 2x2, A2 is 2x1, B1 is 2x3, B2 is 1x3?
        # No, standard block decomposition:
        # A (2x3) = [A_left (2x2) | A_right (2x1)]
        # B (3x3) = [B_top (2x3); B_bottom (1x3)]
        # C = A_left B_top + A_right B_bottom
        # A_left B_top is 2x2 x 2x3. This is not 2x2x2.
        # My previous logic: M_{2,3,2} = M_{2,2,2} + M_{2,1,2}.
        # This implies B is 3x2.
        # Let's stick to the column split of B.
        # B (3x3) = [B_col0 B_col1 B_col2]
        # C = A B_col0 + A B_col1 + A B_col2
        # A B_colj is 2x1.
        # This is 3 blocks of 2x3x1. Rank 6*3 = 18.
        # My previous "17" was based on M_{2,3,2} being 11.
        # M_{2,3,2} corresponds to A(2x3) B(3x2).
        # Split B(3x2) into B_top(2x2) and B_bottom(1x2).
        # C = A_top B_top + A_bottom B_bottom.
        # A_top is A[:, 0:2] (2x2). A_bottom is A[:, 2] (2x1).
        # A_top B_top is 2x2 x 2x2. Rank 7.
        # A_bottom B_bottom is 2x1 x 1x2. Rank 4.
        # Total 11.
        # So M_{2,3,3} = M_{2,3,2} + M_{2,3,1}.
        # M_{2,3,2} uses columns 0,1 of B.
        # M_{2,3,1} uses column 2 of B.
        
        # So for 2x3x3:
        # Block 1 (cols 0,1 of B): Rank 11.
        # Block 2 (col 2 of B): Rank 6.
        
        # I already placed Block 1 (2x2x2 part) in 0:7.
        # I need to place the 2x1x2 part in 7:11.
        # And the 2x3x1 part in 11:17.
        
        U_212, V_212, W_212 = get_2x1x2_rank4()
        # U_212 (2x4). Rows i (0,1).
        # In 2x3x3, A_right is A[:, 2]. So rows 2, 5 of U17.
        # U17[2, 7:11] = U_212[0, :]
        # U17[5, 7:11] = U_212[1, :]
        
        U17[2, 7:11] = U_212[0, :]
        U17[5, 7:11] = U_212[1, :]
        
        # V_212 (2x4). Rows j (0,1).
        # In 2x3x3, B_bottom is B[2, 0:2]. So rows 2*3+0=6, 2*3+1=7 of V17.
        # V17[6, 7:11] = V_212[0, :]
        # V17[7, 7:11] = V_212[1, :]
        
        V17[6, 7:11] = V_212[0, :]
        V17[7, 7:11] = V_212[1, :]
        
        # W_212 (4x4). Rows k*2+i. k is 0,1 (cols 0,1 of C).
        # In 2x3x3, C cols 0,1. So W17 rows 0,1,2,3.
        # W17[0:4, 7:11] = W_212
        
        W17[0, 7:11] = W_212[0, :]
        W17[1, 7:11] = W_212[1, :]
        W17[2, 7:11] = W_212[2, :]
        W17[3, 7:11] = W_212[3, :]
        
        # Block 3: 2x3x1 (col 2 of B)
        U_231, V_231, W_231 = get_2x3x1_rank6()
        # U_231 (6x6). Rows i*3+j.
        # U17[0:6, 11:17] = U_231
        
        U17[:, 11:17] = U_231
        
        # V_231 (3x6). Rows j.
        # In 2x3x3, B col 2 is B[j, 2]. So rows j*3+2 of V17.
        # V17[0:3:3, 11:17] = V_231[0, :]
        # V17[3:6:3, 11:17] = V_231[1, :]
        # V17[6:9:3, 11:17] = V_231[2, :]
        
        V17[2, 11:17] = V_231[0, :]
        V17[5, 11:17] = V_231[1, :]
        V17[8, 11:17] = V_231[2, :]
        
        # W_231 (2x6). Rows i.
        # In 2x3x3, C col 2. So W17 rows 2*2+0=4, 2*2+1=5.
        # W17[4, 11:17] = W_231[0, :]
        # W17[5, 11:17] = W_231[1, :]
        
        W17[4, 11:17] = W_231[0, :]
        W17[5, 11:17] = W_231[1, :]
        
        return U17, V17, W17

    # Construct full 2x3x7
    U17, V17, W17 = get_2x3x3_rank17()
    U6, V6, W6 = get_2x3x1_rank6()
    
    R_total = 17 + 17 + 6
    U_full = np.zeros((N_U, R_total))
    V_full = np.zeros((N_V, R_total))
    W_full = np.zeros((N_W, R_total))
    
    # Block 1: Cols 0-2 of B, Cols 0-2 of C
    U_full[:, 0:17] = U17
    for j in range(3):
        for k in range(3):
            local_idx = j * 3 + k
            global_idx = j * 7 + k
            V_full[global_idx, 0:17] = V17[local_idx, :]
    for k in range(3):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = k * 2 + i
            W_full[global_idx, 0:17] = W17[local_idx, :]
            
    # Block 2: Cols 3-5 of B, Cols 3-5 of C
    U_full[:, 17:34] = U17
    for j in range(3):
        for k in range(3):
            local_idx = j * 3 + k
            global_idx = j * 7 + (k + 3)
            V_full[global_idx, 17:34] = V17[local_idx, :]
    for k in range(3):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = (k + 3) * 2 + i
            W_full[global_idx, 17:34] = W17[local_idx, :]
            
    # Block 3: Col 6 of B, Col 6 of C
    U_full[:, 34:40] = U6
    for j in range(3):
        local_idx = j
        global_idx = j * 7 + 6
        V_full[global_idx, 34:40] = V6[local_idx, :]
    for i in range(2):
        local_idx = i
        global_idx = 6 * 2 + i
        W_full[global_idx, 34:40] = W6[local_idx, :]
        
    # Verify
    T_calc = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    if np.allclose(T_calc, T_target, atol=1e-9):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
        
    return best_result
# EVOLVE-BLOCK-END
