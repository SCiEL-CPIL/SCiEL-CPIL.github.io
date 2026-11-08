import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 22 construction
    # 1. Strassen 2x2x2 (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U_s[0,0] = 1; U_s[3,0] = 1
    V_s[0,0] = 1; V_s[3,0] = 1
    W_s[0,0] = 1; W_s[1,0] = 1; W_s[2,0] = 1; W_s[3,0] = 1
    
    # m2 = (c+d)e
    U_s[2,1] = 1; U_s[3,1] = 1
    V_s[0,1] = 1
    W_s[2,1] = 1
    
    # m3 = a(f-h)
    U_s[0,2] = 1
    V_s[1,2] = 1; V_s[3,2] = -1
    W_s[1,2] = 1
    
    # m4 = c(g-e)
    U_s[2,3] = 1
    V_s[2,3] = 1; V_s[0,3] = -1
    W_s[0,3] = -1; W_s[2,3] = 1
    
    # m5 = (a+b)h
    U_s[0,4] = 1; U_s[1,4] = 1
    V_s[3,4] = 1
    W_s[1,4] = 1; W_s[3,4] = -1
    
    # m6 = (d-b)(f+h)
    U_s[3,5] = 1; U_s[1,5] = -1
    V_s[1,5] = 1; V_s[3,5] = 1
    W_s[0,5] = 1; W_s[3,5] = 1
    
    # m7 = (a-c)(e+g)
    U_s[0,6] = 1; U_s[2,6] = -1
    V_s[0,6] = 1; V_s[2,6] = 1
    W_s[0,6] = 1; W_s[3,6] = -1
    
    # 2. Schoolbook 2x1x2 (Rank 4)
    # A is 2x1 (rows 4,5 of U), B is 1x2 (rows 4,5 of V)
    # C is 2x2
    U_b = np.zeros((2, 4))
    V_b = np.zeros((2, 4))
    W_b = np.zeros((4, 4))
    
    # 1. a*e -> C00
    U_b[0,0] = 1
    V_b[0,0] = 1
    W_b[0,0] = 1
    
    # 2. a*f -> C01
    U_b[0,1] = 1
    V_b[1,1] = 1
    W_b[1,1] = 1
    
    # 3. c*e -> C10
    U_b[1,2] = 1
    V_b[0,2] = 1
    W_b[2,2] = 1
    
    # 4. c*f -> C11
    U_b[1,3] = 1
    V_b[1,3] = 1
    W_b[3,3] = 1
    
    # 3. Combine for 2x3x3 (Rank 11)
    # U: 6x11, V: 6x11, W: 6x11
    U_233 = np.zeros((6, 11))
    V_233 = np.zeros((6, 11))
    W_233 = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_233[0:4, 0:7] = U_s
    V_233[0:4, 0:7] = V_s
    W_233[:, 0:7] = W_s
    
    # Schoolbook part (cols 7-10)
    U_233[4:6, 7:11] = U_b
    V_233[4:6, 7:11] = V_b
    W_233[:, 7:11] = W_b
    
    # 4. Combine for 2x3x6 (Rank 22)
    # U: 6x22, V: 18x22, W: 12x22
    U = np.zeros((6, 22))
    V = np.zeros((18, 22))
    W = np.zeros((12, 22))
    
    # First 3x3 block of B (cols 0-10)
    U[:, 0:11] = U_233
    V[0:6, 0:11] = V_233
    W[0:6, 0:11] = W_233
    
    # Second 3x3 block of B (cols 11-21)
    U[:, 11:22] = U_233
    V[6:12, 11:22] = V_233
    W[6:12, 11:22] = W_233
    
    # Verify
    def verify(U, V, W, n, m, p):
        # U: n*m x R
        # V: m*p x R
        # W: p*n x R
        # C[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = delta[i,j,k]
        # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 if i,j,k match else 0?
        # No, the tensor is T[i,j,k] = 1 if it's a valid multiplication term.
        # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
        # U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] should sum to 1 for the (i,j,k) component of the multiplication tensor.
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
        # T[i,j,k] = 1 if the term A_ij B_jk contributes to C_ik.
        # Actually, T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Wait, the tensor is defined such that C_ik = sum_j A_ij B_jk.
        # The tensor T has T[i,j,k] = 1.
        # So we check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 36
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][column] = 1
                    V_f[j * p + k][column] = 1
                    W_f[k * n + i][column] = 1
                    column += 1
        return {"U": U_f, "V": V_f, "W": W_f}
