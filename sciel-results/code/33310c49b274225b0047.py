import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strassen decomposition for 2x2x2
    def get_strassen():
        n, m, p = 2, 2, 2
        R = 7
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # r=1: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # r=2: (A10+A11)B00 -> C10
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # r=3: A00(B01-B11) -> C01
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # r=4: A11(B10-B00) -> C00, C10
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # r=5: (A00+A01)B11 -> C01, C00(-)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[0, 4] = -1
        
        # r=6: (A10-A00)(B00+B01) -> C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # r=7: (A01-A11)(B10+B11) -> C00
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def direct_sum(U1, V1, W1, U2, V2, W2):
        return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])

    def tensor_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        # U1 is n1*m1 x R1, U2 is n2*m2 x R2
        # New U is (n1*n2)*(m1*m2) x (R1*R2)
        # Index mapping: (i1, i2, j1, j2) -> (i1*n2 + i2)*m1*m2 + j1*m2 + j2
        # Kronecker product of U1 and U2?
        # U1 rows: i1*m1 + j1. U2 rows: i2*m2 + j2.
        # np.kron(U1, U2) has rows (i1*m1+j1)*(m2*m2) + (i2*m2+j2)? No.
        # np.kron(A, B) where A is n1 x n2, B is m1 x m2.
        # Result is (n1*m1) x (n2*m2).
        # Row index: i_A * m1 + i_B.
        # We want row index: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2).
        # This is not standard kron order.
        # Standard kron: A[i1, i2] * B[j1, j2].
        # We need to permute the rows.
        
        U1_exp = U1.reshape(n1, m1, -1)
        U2_exp = U2.reshape(n2, m2, -1)
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        
        U_new = np.zeros((n1*n2*m1*m2, R1*R2))
        
        for i1 in range(n1):
            for i2 in range(n2):
                for j1 in range(m1):
                    for j2 in range(m2):
                        row_new = (i1*n2 + i2)*m1*m2 + j1*m2 + j2
                        for r1 in range(R1):
                            for r2 in range(R2):
                                col_new = r1*R2 + r2
                                U_new[row_new, col_new] = U1_exp[i1, j1, r1] * U2_exp[i2, j2, r2]
                                
        # Similarly for V and W
        V1_exp = V1.reshape(m1, p1, -1)
        V2_exp = V2.reshape(m2, p2, -1)
        V_new = np.zeros((m1*m2*p1*p2, R1*R2))
        for j1 in range(m1):
            for j2 in range(m2):
                for k1 in range(p1):
                    for k2 in range(p2):
                        row_new = (j1*m2 + j2)*p1*p2 + k1*p2 + k2
                        for r1 in range(R1):
                            for r2 in range(R2):
                                col_new = r1*R2 + r2
                                V_new[row_new, col_new] = V1_exp[j1, k1, r1] * V2_exp[j2, k2, r2]
                                
        W1_exp = W1.reshape(p1, n1, -1) # Note: W is p*n, so index is k*n+i
        W2_exp = W2.reshape(p2, n2, -1)
        W_new = np.zeros((p1*p2*n1*n2, R1*R2))
        for k1 in range(p1):
            for k2 in range(p2):
                for i1 in range(n1):
                    for i2 in range(n2):
                        row_new = (k1*p2 + k2)*n1*n2 + i1*n2 + i2
                        for r1 in range(R1):
                            for r2 in range(R2):
                                col_new = r1*R2 + r2
                                W_new[row_new, col_new] = W1_exp[k1, i1, r1] * W2_exp[k2, i2, r2]
                                
        return U_new, V_new, W_new

    # Build M_3,4,8
    # M_2,2,2
    U_s, V_s, W_s = get_strassen()
    
    # M_2,2,4 = M_2,2,2 (col 0-1) + M_2,2,2 (col 2-3)
    # This is direct sum
    U_224, V_224, W_224 = direct_sum(U_s, V_s, W_s, U_s, V_s, W_s)
    
    # M_2,2,8 = M_2,2,4 + M_2,2,4
    U_228, V_228, W_228 = direct_sum(U_224, V_224, W_224, U_224, V_224, W_224)
    
    # M_1,2,2 = M_1,1,2 + M_1,1,2?
    # M_1,1,2: A(1x1) B(1x2) -> C(1x2)
    # Rank 2.
    # U: 1x2, V: 2x2, W: 2x1?
    # Let's just use schoolbook for small ones or construct manually.
    # M_1,1,2:
    # U[0,0]=1, U[0,1]=1
    # V[0,0]=1, V[1,0]=0; V[0,1]=0, V[1,1]=1
    # W[0,0]=1, W[1,0]=0; W[0,1]=0, W[1,1]=1
    # Wait, W is p*n = 2*1 = 2.
    # W[k*n+i, r]. k in 0..1, i in 0..0.
    # W[0,0]=1, W[1,0]=0
    # W[0,1]=0, W[1,1]=1
    
    U_112 = np.array([[1, 1]])
    V_112 = np.array([[1, 0], [0, 1]])
    W_112 = np.array([[1, 0], [0, 1]])
    
    # M_1,2,2 = M_1,1,2 + M_1,1,2
    U_122, V_122, W_122 = direct_sum(U_112, V_112, W_112, U_112, V_112, W_112)
    
    # M_1,2,4 = M_1,2,2 + M_1,2,2
    U_124, V_124, W_124 = direct_sum(U_122, V_122, W_122, U_122, V_122, W_122)
    
    # M_1,2,8 = M_1,2,4 + M_1,2,4
    U_128, V_128, W_128 = direct_sum(U_124, V_124, W_124, U_124, V_124, W_124)
    
    # M_3,4,8 = M_2,4,8 + M_1,4,8
    # M_2,4,8 = M_2,2,8 + M_2,2,8
    U_248, V_248, W_248 = direct_sum(U_228, V_228, W_228, U_228, V_228, W_228)
    
    # M_1,4,8 = M_1,2,8 + M_1,2,8
    U_148, V_148, W_148 = direct_sum(U_128, V_128, W_128, U_128, V_128, W_128)
    
    U_final, V_final, W_final = direct_sum(U_248, V_248, W_248, U_148, V_148, W_148)
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
