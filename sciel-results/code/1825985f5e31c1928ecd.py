import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit

def construct():
    n, m, p = 2, 3, 9
    
    # We need to find a rank-minimal exact tensor decomposition for 2x3 * 3x9
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (27, rank)
    # W: (p*n, rank) = (18, rank)
    # where W[k*n+i, r] corresponds to output C[i, k] (row i, col k)
    
    # Strategy: Use a hybrid approach
    # Part 1: Handle columns 0-7 of B using Strassen for 2x2 blocks (4 blocks of 2 columns)
    # Part 2: Handle column 8 separately
    # Part 3: Handle the third row of A (A[2,:]) * B[2,:]
    
    # Let's build the decomposition
    rank = 50  # Start with same rank, try to verify
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    col_idx = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :] using Strassen for 4 pairs of columns
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices: B[0, k0]=v_e, B[0, k1]=v_f, B[1, k0]=v_g, B[1, k1]=v_h
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # U indices: A[0,0]=u_a, A[0,1]=u_b, A[1,0]=u_c, A[1,1]=u_d
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 3  # A[1,0]
        u_d = 4  # A[1,1]
        
        # Strassen 7 multiplications for 2x2 * 2x2
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
        
        # W indices: output C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # W[k*n+i, r] where i is row, k is col
        w_00 = k0 * n + 0  # C[0, k0]
        w_01 = k1 * n + 0  # C[0, k1]
        w_10 = k0 * n + 1  # C[1, k0]
        w_11 = k1 * n + 1  # C[1, k1]
        
        # C[0,k0] = M1 - M3 + M4 + M7... let me recheck Strassen
        # Standard Strassen:
        # C11 = M1 - M3 + M4 - M7... 
        # Actually let me be more careful:
        # a  b    e  f
        # c  d    g  h
        #
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C11 = M1 - M3 + M4 - M7
        # C12 = M5 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M5
        
        W[w_00][col_idx] += 1    # M1
        W[w_00][col_idx+2] -= 1  # -M3
        W[w_00][col_idx+3] += 1  # +M4
        W[w_00][col_idx+6] -= 1  # -M7
        
        W[w_01][col_idx+4] += 1  # M5
        W[w_01][col_idx+6] += 1  # +M7
        
        W[w_10][col_idx+1] += 1  # M2
        W[w_10][col_idx+3] += 1  # +M4
        
        W[w_11][col_idx] += 1    # M1
        W[w_11][col_idx+1] -= 1  # -M2
        W[w_11][col_idx+2] += 1  # +M3
        W[w_11][col_idx+4] += 1  # +M5
        
        col_idx += 7
    
    # Single column k=8 for A[:, 0:2] * B[0:2, 8]
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    
    u_a = 0; u_b = 1; u_c = 3; u_d = 4
    
    # Direct multiplication for 2x2 * 2x1 (4 mults)
    # C[0,8] = a*e8 + b*g8
    # C[1,8] = c*e8 + d*g8
    
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
    
    # Part 2: A[:, 2] * B[2, :] - 2x1 * 1x9
    # C[i, k] += A[i,2] * B[2,k]
    # This requires 2*9 = 18 multiplications (or we can do 9 if we factor)
    for k in range(9):
        v_b = 2 * p + k
        
        U[2][col_idx] += 1      # A[0,2]
        V[v_b][col_idx] += 1    # B[2,k]
        W[k * n + 0][col_idx] += 1  # C[0,k]
        
        U[5][col_idx+1] += 1    # A[1,2]
        V[v_b][col_idx+1] += 1  # B[2,k]
        W[k * n + 1][col_idx+1] += 1  # C[1,k]
        
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

