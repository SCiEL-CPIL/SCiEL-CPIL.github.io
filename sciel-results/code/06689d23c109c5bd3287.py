"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    # Schoolbook rank is 40. We aim for 36 using block Strassen.
    # Split A (2x4) into A_L (2x2) and A_R (2x2).
    # Split B (4x5) into B_T (2x5) and B_B (2x5).
    # C = A_L B_T + A_R B_B.
    # For each 2x2 * 2x5 product, split 2x5 into two 2x2 and one 2x1.
    # Rank of 2x2*2x2 is 7 (Strassen).
    # Rank of 2x2*2x1 is 4 (Schoolbook).
    # Total rank per 2x2*2x5 is 7 + 7 + 4 = 18.
    # Total rank = 18 + 18 = 36.
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_vals: dict mapping (i, j) to coeff for A[i,j]
    # v_vals: dict mapping (j, k) to coeff for B[j,k]
    # w_vals: dict mapping (k, i) to coeff for C[i,k]
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for (i, j), val in u_vals.items():
            U[i * m + j][col] += val
        for (j, k), val in v_vals.items():
            V[j * p + k][col] += val
        for (k, i), val in w_vals.items():
            W[k * n + i][col] += val
        col += 1
        
    # Strassen for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2x2(a_offset_i, a_offset_j, b_offset_j, b_offset_k):
        # a_offset_i: row offset for A (0 or 2? No, A is 2x4, so i is 0,1)
        # a_offset_j: col offset for A (0 for L, 2 for R)
        # b_offset_j: row offset for B (0 for T, 2 for B)
        # b_offset_k: col offset for B (0, 2, 4)
        
        # M1
        u1 = { (0, a_offset_j): 1, (1, a_offset_j): 1, (0, a_offset_j+1): 0, (1, a_offset_j+1): 0 }
        # Wait, a11 is at (1, a_offset_j+1)
        u1 = { (0, a_offset_j): 1, (1, a_offset_j+1): 1 }
        v1 = { (b_offset_j, b_offset_k): 1, (b_offset_j+1, b_offset_k+1): 1 }
        w1 = { (b_offset_k, 0): 1, (b_offset_k+1, 1): 1 } # C00, C11
        # C00 = M1 ... C11 = M1 ...
        # C00 coeff 1, C11 coeff 1.
        w1 = { (b_offset_k, 0): 1, (b_offset_k+1, 1): 1 }
        add_term(u1, v1, w1)
        
        # M2 = (a10+a11)b00
        u2 = { (1, a_offset_j): 1, (1, a_offset_j+1): 1 }
        v2 = { (b_offset_j, b_offset_k): 1 }
        w2 = { (b_offset_k, 1): 1, (b_offset_k+1, 1): -1 } # C10, C11
        add_term(u2, v2, w2)
        
        # M3 = a00(b10-b11)
        u3 = { (0, a_offset_j): 1 }
        v3 = { (b_offset_j+1, b_offset_k): 1, (b_offset_j+1, b_offset_k+1): -1 }
        w3 = { (b_offset_k+1, 0): 1, (b_offset_k+1, 1): 1 } # C01, C11
        add_term(u3, v3, w3)
        
        # M4 = a11(b01-b00)
        u4 = { (1, a_offset_j+1): 1 }
        v4 = { (b_offset_j, b_offset_k+1): 1, (b_offset_j, b_offset_k): -1 }
        w4 = { (b_offset_k, 0): 1, (b_offset_k, 1): 1 } # C00, C10
        add_term(u4, v4, w4)
        
        # M5 = (a00+a01)b11
        u5 = { (0, a_offset_j): 1, (0, a_offset_j+1): 1 }
        v5 = { (b_offset_j+1, b_offset_k+1): 1 }
        w5 = { (b_offset_k, 0): -1, (b_offset_k+1, 0): 1 } # C00, C01
        add_term(u5, v5, w5)
        
        # M6 = (a10-a00)(b00+b10)
        u6 = { (1, a_offset_j): 1, (0, a_offset_j): -1 }
        v6 = { (b_offset_j, b_offset_k): 1, (b_offset_j+1, b_offset_k): 1 }
        w6 = { (b_offset_k+1, 1): 1 } # C11
        add_term(u6, v6, w6)
        
        # M7 = (a01-a11)(b01+b11)
        u7 = { (0, a_offset_j+1): 1, (1, a_offset_j+1): -1 }
        v7 = { (b_offset_j, b_offset_k+1): 1, (b_offset_j+1, b_offset_k+1): 1 }
        w7 = { (b_offset_k, 0): 1 } # C00
        add_term(u7, v7, w7)

    # Schoolbook for 2x2 * 2x1
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00], [b10]]
    # C00 = a00 b00 + a01 b10
    # C10 = a10 b00 + a11 b10
    
    def add_schoolbook_2x2x1(a_offset_i, a_offset_j, b_offset_j, b_offset_k):
        # Term 1: a00 b00 -> C00
        u1 = { (0, a_offset_j): 1 }
        v1 = { (b_offset_j, b_offset_k): 1 }
        w1 = { (b_offset_k, 0): 1 }
        add_term(u1, v1, w1)
        
        # Term 2: a01 b10 -> C00
        u2 = { (0, a_offset_j+1): 1 }
        v2 = { (b_offset_j+1, b_offset_k): 1 }
        w2 = { (b_offset_k, 0): 1 }
        add_term(u2, v2, w2)
        
        # Term 3: a10 b00 -> C10
        u3 = { (1, a_offset_j): 1 }
        v3 = { (b_offset_j, b_offset_k): 1 }
        w3 = { (b_offset_k, 1): 1 }
        add_term(u3, v3, w3)
        
        # Term 4: a11 b10 -> C10
        u4 = { (1, a_offset_j+1): 1 }
        v4 = { (b_offset_j+1, b_offset_k): 1 }
        w4 = { (b_offset_k, 1): 1 }
        add_term(u4, v4, w4)

    # Process Top Half (A_L * B_T)
    # A_L is A[0:2, 0:2], B_T is B[0:2, 0:5]
    # B_T split into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    add_strassen_2x2x2(0, 0, 0, 0) # A_L * B_T1
    add_strassen_2x2x2(0, 0, 0, 2) # A_L * B_T2
    add_schoolbook_2x2x1(0, 0, 0, 4) # A_L * B_T3
    
    # Process Bottom Half (A_R * B_B)
    # A_R is A[0:2, 2:4], B_B is B[2:4, 0:5]
    # B_B split into B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    add_strassen_2x2x2(0, 2, 2, 0) # A_R * B_B1
    add_strassen_2x2x2(0, 2, 2, 2) # A_R * B_B2
    add_schoolbook_2x2x1(0, 2, 2, 4) # A_R * B_B3
    
    # Verification
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_arr[i * m + j, :]
                    v_row = V_arr[j * p + k, :]
                    w_row = W_arr[k * n + i, :]
                    s = np.dot(np.dot(u_row, v_row), w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
