import time
import numpy as np
import random

WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def check_valid(U, V, W, tol=1e-9):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4

    def build_2x2x3():
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        U2, V2, W2 = build_strassen_2x2_2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                W[idx][r] = W2[idx][r]
        
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1
        
        return U, V, W

    def build_4x4x7_decomposition():
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        U223, V223, W223 = build_2x2x3()
        
        r_total = r4 + 4 * 11
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * 11
                
                for r_local in range(11):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W

    def try_reduce_rank(U, V, W, max_attempts=200):
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
        best_rank = r
        
        for attempt in range(max_attempts):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            if r <= 1:
                break
                
            c1 = random.randint(0, r-1)
            c2 = random.randint(0, r-1)
            
            if c1 == c2:
                continue
            
            for _ in range(5):
                alpha = random.uniform(-2, 2)
                beta = random.uniform(-2, 2)
                
                if abs(alpha) < 1e-6 and abs(beta) < 1e-6:
                    continue
                
                U_new = U_np.copy()
                V_new = V_np.copy()
                W_new = W_np.copy()
                
                U_new[:, c1] = alpha * U_new[:, c1] + beta * U_new[:, c2]
                V_new[:, c1] = alpha * V_new[:, c1] + beta * V_new[:, c2]
                W_new[:, c1] = alpha * W_new[:, c1] + beta * W_new[:, c2]
                
                mask = [i for i in range(r) if i != c2]
                U_new = U_new[:, mask]
                V_new = V_new[:, mask]
                W_new = W_new[:, mask]
                
                r_new = len(U_new[0])
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r_new):
                                s += U_new[i*m + j, c] * V_new[j*p + k, c] * W_new[k*n + i, c]
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid and r_new < best_rank:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = r_new
                    U_np, V_np, W_np = U_new, V_new, W_new
                    r = r_new
                    break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112

    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    if best_rank < 112 and time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_red, V_red, W_red, r_red = try_reduce_rank(
                best_result["U"], best_result["V"], best_result["W"],
                max_attempts=50
            )
            if r_red < best_rank:
                best_result = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = r_red
        except Exception:
            pass
    
    if best_rank < 112 and time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_red2, V_red2, W_red2, r_red2 = try_reduce_rank(
                best_result["U"], best_result["V"], best_result["W"],
                max_attempts=30
            )
            if r_red2 < best_rank:
                best_result = {"U": U_red2, "V": V_red2, "W": W_red2}
                best_rank = r_red2
        except Exception:
            pass

    return best_result
