import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m      # 6
    N_V = m * p      # 21
    N_W = p * n      # 14

    start_time = time.time()
    deadline = 120.0

    def get_2x3x2_rank11():
        """
        Decompose 2x3x2 matrix multiplication (A is 2x3, B is 3x2, C is 2x2)
        into rank 11 using Strassen for the 2x2 part and rank-4 for the third column.
        """
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Strassen for A[:, 0:2] * B[:, :] -> C
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        # M2 = (A21+A22)B11
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        # M3 = A11(B21-B22)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        # M4 = A22(B12-B11)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        # M5 = (A11+A12)B22
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        # M6 = (A21-A11)(B11+B12)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        # M7 = (A12-A22)(B21+B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        U[0, 0:7] = U_s[0, :]
        U[1, 0:7] = U_s[1, :]
        U[3, 0:7] = U_s[2, :]
        U[4, 0:7] = U_s[3, :]
        
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[2, 0:7] = V_s[2, :]
        V[3, 0:7] = V_s[3, :]
        
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        
        # Rank-4 for A[:, 2] * B[:, :] -> C
        U_212 = np.zeros((2, 4))
        V_212 = np.zeros((2, 4))
        W_212 = np.zeros((4, 4))
        
        U_212[0, 0] = 1; V_212[0, 0] = 1; W_212[0, 0] = 1
        U_212[0, 1] = 1; V_212[1, 1] = 1; W_212[1, 1] = 1
        U_212[1, 2] = 1; V_212[0, 2] = 1; W_212[2, 2] = 1
        U_212[1, 3] = 1; V_212[1, 3] = 1; W_212[3, 3] = 1
        
        U[2, 7:11] = U_212[0, :]
        U[5, 7:11] = U_212[1, :]
        V[4, 7:11] = V_212[0, :]
        V[5, 7:11] = V_212[1, :]
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        
        return U, V, W

    def get_2x3x1_rank3():
        """
        Decompose 2x3x1 matrix multiplication into rank 3.
        C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        """
        U = np.zeros((6, 3))
        V = np.zeros((3, 3))
        W = np.zeros((2, 3))
        for j in range(3):
            U[j, j] = 1
            V[j, j] = 1
            W[0, j] = 1
        for j in range(3):
            U[3+j, j] = 1
            W[1, j] = 1
        return U, V, W

    def build_2x3x7():
        """
        Build the full 2x3x7 decomposition using:
        - Block 1: B cols 0-1, C cols 0-1 (2x3x2, rank 11)
        - Block 2: B cols 2-3, C cols 2-3 (2x3x2, rank 11)
        - Block 3: B cols 4-5, C cols 4-5 (2x3x2, rank 11)
        - Block 4: B col 6, C col 6 (2x3x1, rank 3)
        Total rank: 36
        """
        U11_a, V11_a, W11_a = get_2x3x2_rank11()
        U11_b, V11_b, W11_b = get_2x3x2_rank11()
        U11_c, V11_c, W11_c = get_2x3x2_rank11()
        U3, V3, W3 = get_2x3x1_rank3()
        
        R_total = 36
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1
        U_full[:, 0:11] = U11_a
        for j in range(3):
            for k in range(2):
                V_full[j*7+k, 0:11] = V11_a[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:11] = W11_a[k*2+i, :]
                
        # Block 2
        U_full[:, 11:22] = U11_b
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+2), 11:22] = V11_b[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+2)*2+i, 11:22] = W11_b[k*2+i, :]
                
        # Block 3
        U_full[:, 22:33] = U11_c
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+4), 22:33] = V11_c[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+4)*2+i, 22:33] = W11_c[k*2+i, :]
                
        # Block 4: B col 6, C col 6
        U_full[:, 33:36] = U3
        for j in range(3):
            V_full[j*7+6, 33:36] = V3[j, :]
        for i in range(2):
            W_full[6*2+i, 33:36] = W3[i, :]
            
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

    # Try the rank-36 decomposition
    U_full, V_full, W_full = build_2x3x7()

    if verify(U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }

    # Fallback to schoolbook
    return get_schoolbook()
