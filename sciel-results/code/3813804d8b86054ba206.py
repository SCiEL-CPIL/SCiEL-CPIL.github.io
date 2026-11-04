import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to build a rank-57 decomposition using block decomposition
    # C = A_a * B_a + A_b * B_b
    # A_a: 2x2 (cols 0,1), B_a: 2x6 (rows 0,1) -> use Strassen, rank 21
    # A_b: 2x3 (cols 2,3,4), B_b: 3x6 (rows 2,3,4) -> schoolbook, rank 36
    
    # First, let's try to construct and verify a rank-7 decomposition for 2x2x2
    # If it works, we can use it for the 2x2x6 part
    
    def build_2x2x2_strassen():
        """
        Build rank-7 decomposition for 2x2 * 2x2.
        U: (4,7), V: (4,7), W: (4,7)
        U index: i*2+j, V index: j*2+k, W index: k*2+i
        """
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # M1 = (a+d)(e+h)
        # a=U[0], d=U[3], e=V[0], h=V[3]
        # Contributes to C[0,0]=W[0] and C[1,1]=W[3]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)(e+f)
        # c=U[2], d=U[3], e=V[0], f=V[1]
        # Contributes to C[1,0]=W[1] and C[1,1]=W[3]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1; V[1, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        # a=U[0], f=V[1], h=V[3]
        # Contributes to C[0,1]=W[2] and C[1,1]=W[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        # d=U[3], g=V[2], e=V[0]
        # Contributes to C[0,0]=W[0] and C[1,0]=W[1]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a+b)h
        # a=U[0], b=U[1], h=V[3]
        # Contributes to C[0,0]=W[0] and C[0,1]=W[2]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (c-a)(e+g)
        # c=U[2], a=U[0], e=V[0], g=V[2]
        # Contributes to C[1,1]=W[3]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h)
        # b=U[1], d=U[3], g=V[2], h=V[3]
        # Contributes to C[0,0]=W[0]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Verify the 2x2x2 Strassen decomposition
    U_str, V_str, W_str = build_2x2x2_strassen()
    
    # Test: for all i,j,k in {0,1}, sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] should be 1
    strassen_valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u_idx = i * 2 + j
                v_idx = j * 2 + k
                w_idx = k * 2 + i
                val = np.sum(U_str[u_idx, :] * V_str[v_idx, :] * W_str[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    strassen_valid = False
    
    if strassen_valid:
        # Build the full rank-57 decomposition
        R_total = 21 + 36
        U = np.zeros((10, R_total))
        V = np.zeros((30, R_total))
        W = np.zeros((12, R_total))
        
        # Strassen part for 2x2 * 2x6
        # Split B (2x6) into three 2x2 blocks
        for t in range(3):
            base = t * 7
            # U_str rows map to U rows: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
            U[0, base:base+7] = U_str[0, :]
            U[1, base:base+7] = U_str[1, :]
            U[5, base:base+7] = U_str[2, :]
            U[6, base:base+7] = U_str[3, :]
            
            # V_str rows map to V rows for block t
            # V_str row 0 -> B[0, 2*t] -> V index 2*t
            # V_str row 1 -> B[0, 2*t+1] -> V index 2*t+1
            # V_str row 2 -> B[1, 2*t] -> V index 6+2*t
            # V_str row 3 -> B[1, 2*t+1] -> V index 7+2*t
            V[2*t, base:base+7] = V_str[0, :]
            V[2*t+1, base:base+7] = V_str[1, :]
            V[6+2*t, base:base+7] = V_str[2, :]
            V[7+2*t, base:base+7] = V_str[3, :]
            
            # W_str rows map to W rows for block t
            # W_str row 0 -> C[0, 2*t] -> W index (2*t)*2+0 = 4*t
            # W_str row 1 -> C[1, 2*t] -> W index (2*t)*2+1 = 4*t+1
            # W_str row 2 -> C[0, 2*t+1] -> W index (2*t+1)*2+0 = 4*t+2
            # W_str row 3 -> C[1, 2*t+1] -> W index (2*t+1)*2+1 = 4*t+3
            W[4*t, base:base+7] = W_str[0, :]
            W[4*t+1, base:base+7] = W_str[1, :]
            W[4*t+2, base:base+7] = W_str[2, :]
            W[4*t+3, base:base+7] = W_str[3, :]
        
        # Schoolbook part for 2x3 * 3x6
        col = 21
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    u_idx = i * 5 + (j + 2)  # columns 2,3,4 of A
                    v_idx = (j + 2) * 6 + k  # rows 2,3,4 of B
                    w_idx = k * 2 + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

