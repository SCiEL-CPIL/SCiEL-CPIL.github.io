import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s limit

def construct():
    n, m, p = 2, 3, 9
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    # Part 1: A[:, 0:2] * B[0:2, :] using Strassen for 4 pairs + 1 direct
    col_idx = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0  # A00
        u_b = 1  # A01
        u_c = 3  # A10
        u_d = 4  # A11
        
        # M1 = (a+d)(e+h)
        U[u_a][col_idx] += 1; U[u_d][col_idx] += 1
        V[v_e][col_idx] += 1; V[v_h][col_idx] += 1
        
        # M2 = (c+d)e
        U[u_c][col_idx+1] += 1; U[u_d][col_idx+1] += 1
        V[v_e][col_idx+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col_idx+2] += 1
        V[v_f][col_idx+2] += 1; V[v_h][col_idx+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col_idx+3] += 1
        V[v_g][col_idx+3] += 1; V[v_e][col_idx+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col_idx+4] += 1; U[u_b][col_idx+4] += 1
        V[v_h][col_idx+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col_idx+5] += 1; U[u_a][col_idx+5] -= 1
        V[v_e][col_idx+5] += 1; V[v_f][col_idx+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col_idx+6] += 1; U[u_d][col_idx+6] -= 1
        V[v_g][col_idx+6] += 1; V[v_h][col_idx+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col_idx] += 1
        W[w_00][col_idx+3] += 1
        W[w_00][col_idx+4] -= 1
        W[w_00][col_idx+6] += 1
        
        W[w_01][col_idx+2] += 1
        W[w_01][col_idx+4] += 1
        
        W[w_10][col_idx+1] += 1
        W[w_10][col_idx+3] += 1
        
        W[w_11][col_idx] += 1
        W[w_11][col_idx+1] -= 1
        W[w_11][col_idx+2] += 1
        W[w_11][col_idx+5] += 1
        
        col_idx += 7
    
    # Single column k=8 for A[:, 0:2] * B[0:2, 8]
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    
    u_a = 0; u_b = 1; u_c = 3; u_d = 4
    
    # Direct multiplication for 2x2 x 2x1
    U[u_a][col_idx] += 1
    V[v_e8][col_idx] += 1
    W[k8 * n + 0][col_idx] += 1
    
    U[u_b][col_idx+1] += 1
    V[v_g8][col_idx+1] += 1
    W[k8 * n + 0][col_idx+1] += 1
    
    U[u_c][col_idx+2] += 1
    V[v_e8][col_idx+2] += 1
    W[k8 * n + 1][col_idx+2] += 1
    
    U[u_d][col_idx+3] += 1
    V[v_g8][col_idx+3] += 1
    W[k8 * n + 1][col_idx+3] += 1
    
    col_idx += 4
    
    # Part 2: A[:, 2:3] * B[2, :]
    for k in range(9):
        v_b = 2 * p + k
        
        U[2][col_idx] += 1
        V[v_b][col_idx] += 1
        W[k * n + 0][col_idx] += 1
        
        U[5][col_idx+1] += 1
        V[v_b][col_idx+1] += 1
        W[k * n + 1][col_idx+1] += 1
        
        col_idx += 2

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 3, 9
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    C_flat = np.zeros(n * p)
    U_arr = np.array(res["U"])
    V_arr = np.array(res["V"])
    W_arr = np.array(res["W"])
    
    for s in range(len(U_arr[0])):
        val = np.dot(U_arr[:, s], A_flat) * np.dot(V_arr[:, s], B_flat)
        C_flat += val * W_arr[:, s]
    
    C_actual = C_flat.reshape((p, n)).T
    
    if not np.allclose(C_actual, C_expected, atol=1e-6):
        print("Mismatch detected")
    else:
        print("Validation Passed")
