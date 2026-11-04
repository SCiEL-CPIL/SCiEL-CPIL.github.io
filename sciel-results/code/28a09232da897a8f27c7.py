import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m      # 6
    N_V = m * p      # 21
    N_W = p * n      # 14

    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s limit

    # Build a rank-39 decomposition using Strassen for 2x2x2 blocks
    # Structure: 3 blocks of 2x3x2 (rank 11 each) + 1 block of 2x3x1 (rank 6) = 39

    def get_2x3x2_rank11():
        """
        Decompose 2x3x2 matrix multiplication (A is 2x3, B is 3x2, C is 2x2)
        into rank 11 using Strassen for the 2x2 part (first two columns of A)
        and rank-4 for the remaining row of A.
        
        Returns:
            U: (6, 11) - rows for A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
            V: (6, 11) - rows for B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1]
            W: (4, 11) - rows for C[0,0], C[0,1], C[1,0], C[1,1]
        """
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Part 1: Strassen 2x2x2 for A[:, 0:2] * B[:, :] -> C
        # This handles the contribution from A[0,0], A[0,1], A[1,0], A[1,1]
        # Using standard Strassen decomposition
        U_s = np.zeros((4, 7))  # for A[0,0], A[0,1], A[1,0], A[1,1]
        V_s = np.zeros((4, 7))  # for B[0,0], B[0,1], B[1,0], B[1,1]
        W_s = np.zeros((4, 7))  # for C[0,0], C[0,1], C[1,0], C[1,1]
        
        # Strassen formulas
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
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
        
        # Map Strassen to full U, V, W
        # U rows: A[0,0](0), A[0,1](1), A[0,2](2), A[1,0](3), A[1,1](4), A[1,2](5)
        U[0, 0:7] = U_s[0, :]  # A[0,0]
        U[1, 0:7] = U_s[1, :]  # A[0,1]
        U[3, 0:7] = U_s[2, :]  # A[1,0]
        U[4, 0:7] = U_s[3, :]  # A[1,1]
        
        # V rows: B[0,0](0), B[0,1](1), B[1,0](2), B[1,1](3), B[2,0](4), B[2,1](5)
        V[0, 0:7] = V_s[0, :]  # B[0,0]
        V[1, 0:7] = V_s[1, :]  # B[0,1]
        V[2, 0:7] = V_s[2, :]  # B[1,0]
        V[3, 0:7] = V_s[3, :]  # B[1,1]
        
        # W rows: C[0,0](0), C[0,1](1), C[1,0](2), C[1,1](3)
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        
        # Part 2: Rank-4 for A[:, 2] * B[:, :] -> C
        # This handles the contribution from A[0,2], A[1,2]
        # C[i,k] += A[i,2] * B[2,k]
        U_212 = np.zeros((2, 4))  # for A[0,2], A[1,2]
        V_212 = np.zeros((2, 4))  # for B[2,0], B[2,1]
        W_212 = np.zeros((4, 4))  # for C[0,0], C[0,1], C[1,0], C[1,1]
        
        # Direct rank-4 decomposition
        U_212[0, 0] = 1; V_212[0, 0] = 1; W_212[0, 0] = 1
        U_212[0, 1] = 1; V_212[1, 1] = 1; W_212[1, 1] = 1
        U_212[1, 2] = 1; V_212[0, 2] = 1; W_212[2, 2] = 1
        U_212[1, 3] = 1; V_212[1, 3] = 1; W_212[3, 3] = 1
        
        U[2, 7:11] = U_212[0, :]  # A[0,2]
        U[5, 7:11] = U_212[1, :]  # A[1,2]
        V[4, 7:11] = V_212[0, :]  # B[2,0]
        V[5, 7:11] = V_212[1, :]  # B[2,1]
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        
        return U, V, W

    def get_2x3x1_rank6():
        """
        Decompose 2x3x1 matrix multiplication (A is 2x3, B is 3x1, C is 2x1)
        into rank 6 using schoolbook.
        
        Returns:
            U: (6, 6) - rows for A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
            V: (3, 6) - rows for B[0,0], B[1,0], B[2,0]
            W: (2, 6) - rows for C[0,0], C[1,0]
        """
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
        """
        Build the full 2x3x7 decomposition by combining:
        - Block 1: B cols 0-1, C cols 0-1 (2x3x2, rank 11)
        - Block 2: B cols 2-3, C cols 2-3 (2x3x2, rank 11)
        - Block 3: B cols 4-5, C cols 4-5 (2x3x2, rank 11)
        - Block 4: B col 6, C col 6 (2x3x1, rank 6)
        Total rank: 39
        """
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
        """
        Verify that U, V, W form a valid tensor decomposition.
        """
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    def get_schoolbook():
        """Fallback: schoolbook rank-42 decomposition."""
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

    # Try the rank-39 decomposition
    U_full, V_full, W_full = build_2x3x7()

    if verify(U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    # Fallback to schoolbook
    return get_schoolbook()
