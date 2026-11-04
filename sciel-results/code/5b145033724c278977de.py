import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 130.0  # 20s safety margin from 150s limit

    # Schoolbook rank: 2*3*7 = 42
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

    # 2x3x2 rank-11 decomposition
    # Combines Strassen 2x2x2 (rank 7) for first two cols and 2x1x2 (rank 4) for last col?
    # Actually, the seed used a specific construction. I will replicate the seed's logic for 2x3x2.
    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        
        # Strassen 2x2x2 part (cols 0,1 of B, cols 0,1 of C)
        # U_s, V_s, W_s are 4x7
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        # M2
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        # M3
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        # M4
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        # M5
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        # M6
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        # M7
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        # Map Strassen to 2x3x2 U, V, W
        # U rows: A[0,0], A[0,1], A[1,0], A[1,1], A[0,2], A[1,2]
        # Strassen uses A[0,0], A[0,1], A[1,0], A[1,1]
        U[0, 0:7] = U_s[0, :]
        U[1, 0:7] = U_s[1, :]
        U[3, 0:7] = U_s[2, :]
        U[4, 0:7] = U_s[3, :]
        
        # V rows: B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1], B[0,2], B[1,2], B[2,2]
        # Strassen uses B[0,0], B[0,1], B[1,0], B[1,1] (indices 0,1,2,3 in V_s)
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[3, 0:7] = V_s[2, :]
        V[4, 0:7] = V_s[3, :]
        
        # W rows: C[0,0], C[0,1], C[1,0], C[1,1]
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        
        # 2x1x2 part for col 2 of B and col 2 of C?
        # Wait, the seed used 2x1x2 for the 3rd column of the 2x3x2 block?
        # The block is 2x3x2. B is 3x2. Cols 0,1.
        # So Strassen handles B cols 0,1 and C cols 0,1.
        # What about B col 2? It doesn't exist in 2x3x2.
        # Ah, the seed's get_2x3x2_rank11 handled B cols 0,1,2?
        # No, 2x3x2 means A is 2x3, B is 3x2.
        # So B has 2 columns.
        # Strassen handles 2x2 part of B (cols 0,1) and 2x2 part of A (rows 0,1).
        # But A is 2x3. Row 2 of A is unused in Strassen.
        # So we need to handle A[0,2] and A[1,2] multiplied by B[:, 0] and B[:, 1].
        # This is a 2x1x2 multiplication? No, it's 2x1 (from A) by 1x2 (from B)?
        # No, A[0,2] and A[1,2] are scalars. B[:, 0] and B[:, 1] are vectors.
        # The product contributes to C[:, 0] and C[:, 1].
        # Specifically, C[i, k] += A[i, 2] * B[2, k].
        # This is a rank-4 addition?
        # Let's check the seed's code for get_2x3x2_rank11.
        # It used U_212, V_212, W_212.
        # U_212 was 2x4. V_212 was 2x4. W_212 was 4x4.
        # It mapped U[2] and U[5] (A[0,2] and A[1,2]) to U_212.
        # It mapped V[6], V[7], V[8] (B[0,2], B[1,2], B[2,2])?
        # Wait, in 2x3x2, B is 3x2. Indices are 0..8.
        # B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4, B[2,0]=6, B[2,1]=7.
        # There is no B[:, 2].
        # The seed's code for get_2x3x2_rank11 seems to have handled a 2x3x3 block?
        # Let's re-read the seed's get_2x3x2_rank11.
        # It says: "2x1x2 handles col 2 of B and col 2 of C".
        # This implies the block was actually 2x3x3?
        # No, the function name is get_2x3x2_rank11.
        # But the code inside uses V_212 with 2 rows?
        # V_212 was 2x4.
        # V_full[6, 7:11] = V_212[0, :] # B[0,2]
        # V_full[7, 7:11] = V_212[1, :] # B[1,2]
        # This implies B had a column 2.
        # So the seed's "2x3x2" was actually a 2x3x3 block truncated?
        # Or did it handle the 3rd column of A?
        # In 2x3x2, A is 2x3. B is 3x2.
        # The product is 2x2.
        # The term A[i,2] B[2,k] contributes to C[i,k].
        # This is a rank-4 tensor (2x1x2).
        # So we add rank 4 to the Strassen rank 7? Total 11.
        # This matches.
        
        # Let's implement this 2x1x2 part for the 3rd row of A.
        # U rows for A[0,2] (index 2) and A[1,2] (index 5).
        # V rows for B[2,0] (index 6) and B[2,1] (index 7).
        # W rows for C[0,0] (0), C[0,1] (1), C[1,0] (2), C[1,1] (3).
        
        # 2x1x2 decomposition (rank 4)
        # A2 is 2x1, B2 is 1x2. C2 is 2x2.
        # C2[i,k] = A2[i] B2[k].
        # U_212[0, :] corresponds to A2[0].
        # U_212[1, :] corresponds to A2[1].
        # V_212[0, :] corresponds to B2[0].
        # V_212[1, :] corresponds to B2[1].
        # W_212[0, :] corresponds to C2[0,0].
        # W_212[1, :] corresponds to C2[0,1].
        # W_212[2, :] corresponds to C2[1,0].
        # W_212[3, :] corresponds to C2[1,1].
        
        U_212 = np.zeros((2, 4))
        V_212 = np.zeros((2, 4))
        W_212 = np.zeros((4, 4))
        
        # Rank 4 for 2x1x2
        U_212[0, 0] = 1; V_212[0, 0] = 1; W_212[0, 0] = 1
        U_212[0, 1] = 1; V_212[1, 1] = 1; W_212[1, 1] = 1
        U_212[1, 2] = 1; V_212[0, 2] = 1; W_212[2, 2] = 1
        U_212[1, 3] = 1; V_212[1, 3] = 1; W_212[3, 3] = 1
        
        U[2, 7:11] = U_212[0, :]
        U[5, 7:11] = U_212[1, :]
        V[6, 7:11] = V_212[0, :]
        V[7, 7:11] = V_212[1, :]
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        
        return U, V, W

    # 2x3x1 rank-6
    def get_2x3x1_rank6():
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

    def build_2x3x7():
        U11_a, V11_a, W11_a = get_2x3x2_rank11()
        U11_b, V11_b, W11_b = get_2x3x2_rank11()
        U11_c, V11_c, W11_c = get_2x3x2_rank11()
        U6, V6, W6 = get_2x3x1_rank6()
        R_total = 39
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1: B cols 0-1, C cols 0-1
        U_full[:, 0:11] = U11_a
        for j in range(3):
            for k in range(2):
                V_full[j*7+k, 0:11] = V11_a[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:11] = W11_a[k*2+i, :]
                
        # Block 2: B cols 2-3, C cols 2-3
        U_full[:, 11:22] = U11_b
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+2), 11:22] = V11_b[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+2)*2+i, 11:22] = W11_b[k*2+i, :]
                
        # Block 3: B cols 4-5, C cols 4-5
        U_full[:, 22:33] = U11_c
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+4), 22:33] = V11_c[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+4)*2+i, 22:33] = W11_c[k*2+i, :]
                
        # Block 4: B col 6, C col 6
        U_full[:, 33:39] = U6
        for j in range(3):
            V_full[j*7+6, 33:39] = V6[j, :]
        for i in range(2):
            W_full[6*2+i, 33:39] = W6[i, :]
            
        return U_full, V_full, W_full

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    U_full, V_full, W_full = build_2x3x7()

    if verify(U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    # Fallback
    return get_schoolbook()
