import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2  # 8 blocks of 2x2
    R = num_blocks * 7   # rank = 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen coefficients for 2x2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B local indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C local indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # U_coeffs[s][a_idx]
    U_coeffs = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    # V_coeffs[s][b_local_idx]
    V_coeffs = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    # W_contribs[s][c_local_idx]
    W_contribs = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, C11
        [0, 1, 0, 1],   # M3 -> C01, C11
        [1, 0, 1, 0],   # M4 -> C00, C10
        [-1, 1, 0, 0],  # M5 -> C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0]    # M7 -> C00
    ]

    for q in range(num_blocks):
        # Global indices for B in block q (B is m x p = 2x16, row-major: j*p+k)
        b_00 = 0 * p + 2 * q       # B[0, 2q]
        b_01 = 0 * p + (2 * q + 1) # B[0, 2q+1]
        b_10 = 1 * p + 2 * q       # B[1, 2q]
        b_11 = 1 * p + (2 * q + 1) # B[1, 2q+1]
        
        # Global indices for C in block q (C is n x p = 2x16, convention: k*n+i)
        c_00 = (2 * q) * n + 0     # C[0, 2q]
        c_01 = (2 * q + 1) * n + 0 # C[0, 2q+1]
        c_10 = (2 * q) * n + 1     # C[1, 2q]
        c_11 = (2 * q + 1) * n + 1 # C[1, 2q+1]
        
        b_map = [b_00, b_01, b_10, b_11]
        c_map = [c_00, c_01, c_10, c_11]
        
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for a_idx in range(4):
                U[a_idx, r] = U_coeffs[s][a_idx]
                
            # Fill V
            for local_idx in range(4):
                V[b_map[local_idx], r] = V_coeffs[s][local_idx]
                
            # Fill W
            for local_out_idx in range(4):
                W[c_map[local_out_idx], r] = W_contribs[s][local_out_idx]

    # Verification using tensor identity: sum_r U[u,r]*V[v,r]*W[w,r] = 1 for valid (u,v,w)
    if time.time() < deadline - 5:
        is_valid = True
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    val = np.dot(U[u, :] * V[v, :], W[w, :])
                    if abs(val - 1.0) > 1e-9:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break

        if not is_valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
