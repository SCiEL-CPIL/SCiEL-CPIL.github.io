import numpy as np
import time

DIMENSIONS = [2, 3, 7]

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
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_2x1x2_rank4():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

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

    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_212, V_212, W_212 = get_2x1x2_rank4()
        
        U[0, 0:7] = U_s[0, :]
        U[1, 0:7] = U_s[1, :]
        U[3, 0:7] = U_s[2, :]
        U[4, 0:7] = U_s[3, :]
        
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[3, 0:7] = V_s[2, :]
        V[4, 0:7] = V_s[3, :]
        
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        
        U[2, 7:11] = U_212[0, :]
        U[5, 7:11] = U_212[1, :]
        
        V[6, 7:11] = V_212[0, :]
        V[7, 7:11] = V_212[1, :]
        
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        
        return U, V, W

    def get_2x3x3_rank17():
        U17 = np.zeros((6, 17))
        V17 = np.zeros((9, 17))
        W17 = np.zeros((6, 17))
        
        U11, V11, W11 = get_2x3x2_rank11()
        U6, V6, W6 = get_2x3x1_rank6()
        
        U17[:, 0:11] = U11
        V17[:, 0:11] = V11
        W17[0:4, 0:11] = W11
        
        U17[:, 11:17] = U6
        V17[2, 11:17] = V6[0, :]
        V17[5, 11:17] = V6[1, :]
        V17[8, 11:17] = V6[2, :]
        W17[4, 11:17] = W6[0, :]
        W17[5, 11:17] = W6[1, :]
        
        return U17, V17, W17

    # Construct full 2x3x7 using 11 + 11 + 11 + 6 = 39
    U11_a, V11_a, W11_a = get_2x3x2_rank11()
    U11_b, V11_b, W11_b = get_2x3x2_rank11()
    U11_c, V11_c, W11_c = get_2x3x2_rank11()
    U6, V6, W6 = get_2x3x1_rank6()
    
    R_total = 11 + 11 + 11 + 6
    U_full = np.zeros((N_U, R_total))
    V_full = np.zeros((N_V, R_total))
    W_full = np.zeros((N_W, R_total))
    
    # Block 1: Cols 0-1 of B, Cols 0-1 of C
    U_full[:, 0:11] = U11_a
    for j in range(3):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + k
            V_full[global_idx, 0:11] = V11_a[local_idx, :]
    for k in range(2):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = k * 2 + i
            W_full[global_idx, 0:11] = W11_a[local_idx, :]
            
    # Block 2: Cols 2-3 of B, Cols 2-3 of C
    U_full[:, 11:22] = U11_b
    for j in range(3):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + (k + 2)
            V_full[global_idx, 11:22] = V11_b[local_idx, :]
    for k in range(2):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = (k + 2) * 2 + i
            W_full[global_idx, 11:22] = W11_b[local_idx, :]
            
    # Block 3: Cols 4-5 of B, Cols 4-5 of C
    U_full[:, 22:33] = U11_c
    for j in range(3):
        for k in range(2):
            local_idx = j * 2 + k
            global_idx = j * 7 + (k + 4)
            V_full[global_idx, 22:33] = V11_c[local_idx, :]
    for k in range(2):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = (k + 4) * 2 + i
            W_full[global_idx, 22:33] = W11_c[local_idx, :]
            
    # Block 4: Col 6 of B, Col 6 of C
    U_full[:, 33:39] = U6
    for j in range(3):
        local_idx = j
        global_idx = j * 7 + 6
        V_full[global_idx, 33:39] = V6[local_idx, :]
    for i in range(2):
        local_idx = i
        global_idx = 6 * 2 + i
        W_full[global_idx, 33:39] = W6[local_idx, :]
        
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
