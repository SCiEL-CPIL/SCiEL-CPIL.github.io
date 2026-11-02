"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen decomposition for 2x2x2 matrix multiplication.
    # 7 products M_s (s=0..6), each with coefficients in A, B, C.
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = A11(B00-B10)  
    # M2 = (A00-A10)(B10+B11)
    # M3 = A00(B01-B11)
    # M4 = (A01+A10)B00
    # M5 = (A11-A01)(B10+B11)
    # M6 = (A00-A10)(B00+B01)
    
    # C00 = M0 + M2 - M4 - M6
    # C01 = M3 + M5
    # C10 = M1 + M4
    # C11 = M0 - M1 + M3 + M5
    
    # Coefficients: A_s[i*2+j], B_s[i*2+j], C_s[i*2+j] for each product s
    A_coeffs = [
        {0: 1, 3: 1},   # M0: A00 + A11
        {3: 1},          # M1: A11
        {0: 1, 2: -1},   # M2: A00 - A10
        {0: 1},          # M3: A00
        {1: 1, 2: 1},   # M4: A01 + A10
        {3: 1, 1: -1},   # M5: A11 - A01
        {0: 1, 2: -1},   # M6: A00 - A10
    ]
    B_coeffs = [
        {0: 1, 3: 1},   # M0: B00 + B11
        {0: 1, 2: -1},   # M1: B00 - B10
        {2: 1, 3: 1},   # M2: B10 + B11
        {1: 1, 3: -1},   # M3: B01 - B11
        {0: 1},          # M4: B00
        {2: 1, 3: 1},   # M5: B10 + B11
        {0: 1, 1: 1},   # M6: B00 + B01
    ]
    C_coeffs = [
        {0: 1, 2: -1, 1: -1, 3: -1},  # M0 -> C00(+1), C01(-1), C10(-1), C11(-1)
        {2: 1, 3: -1},                # M1 -> C10(+1), C11(-1)
        {0: 1},                       # M2 -> C00(+1)
        {1: 1, 3: 1},                # M3 -> C01(+1), C11(+1)
        {2: 1},                       # M4 -> C10(+1)
        {1: 1, 3: 1},                # M5 -> C01(+1), C11(+1)
        {0: -1},                      # M6 -> C00(-1)
    ]
    
    # Verify Strassen locally before embedding
    def verify_strassen_local():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Tensor entry should be 1
                    val = 0
                    for s in range(7):
                        a_idx = i * 2 + j
                        b_idx = j * 2 + k
                        c_idx = k * 2 + i
                        a_c = A_coeffs[s].get(a_idx, 0)
                        b_c = B_coeffs[s].get(b_idx, 0)
                        c_c = C_coeffs[s].get(c_idx, 0)
                        val += a_c * b_c * c_c
                    if val != 1:
                        print(f"Strassen local fail: i={i},j={j},k={k},val={val}")
                        return False
        return True
    
    if not verify_strassen_local():
        # Fallback to schoolbook rank-24
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][idx] = 1
                    V_fb[j * p + k][idx] = 1
                    W_fb[k * n + i][idx] = 1
                    idx += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Embed 3 Strassen blocks (one for each pair of B columns)
    for r in range(3):  # block r handles B columns 2r, 2r+1
        for s in range(7):  # Strassen product s
            col = r * 7 + s
            
            # Fill U: A indices (i*2+j) -> U[i*m+j][col]
            for a_idx, val in A_coeffs[s].items():
                i = a_idx // 2
                j = a_idx % 2
                U[i * m + j][col] = val
            
            # Fill V: B indices (j*2+kl) -> V[j*p + (2*r+kl)][col]
            for b_idx, val in B_coeffs[s].items():
                j = b_idx // 2
                kl = b_idx % 2
                global_k = 2 * r + kl
                V[j * p + global_k][col] = val
            
            # Fill W: C indices (k*2+i) -> W[(2*r+k)*n + i][col]
            for c_idx, val in C_coeffs[s].items():
                k = c_idx // 2
                i = c_idx % 2
                global_k = 2 * r + k
                W[global_k * n + i][col] = val
    
    # Full verification with numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_full(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_arr[u_idx], W_arr[w_idx] * V_arr[v_idx])
                    if abs(val - 1.0) > 1e-9:
                        print(f"Verify fail: i={i},j={j},k={k},val={val}")
                        return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook rank-24
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][idx] = 1
                    V_fb[j * p + k][idx] = 1
                    W_fb[k * n + i][idx] = 1
                    idx += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
