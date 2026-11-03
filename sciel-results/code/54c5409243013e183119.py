"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Using Strassen on 2x2 blocks of 3x3 matrices
    # Rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen matrices for 2x2 blocks
    # Indices: block_row, block_col (0 or 1)
    # U_S[b_i, b_j, k], V_S[b_j, b_k, k], W_S[b_k, b_i, k]
    # k is 0..6
    
    # Initialize Strassen coefficients
    U_S = [[0]*7 for _ in range(4)] # 4 entries: 00, 01, 10, 11
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    def idx(b1, b2):
        return b1 * 2 + b2

    # M1: (A00 + A11)(B00 + B11)
    U_S[idx(0,0)][0] = 1; U_S[idx(1,1)][0] = 1
    V_S[idx(0,0)][0] = 1; V_S[idx(1,1)][0] = 1
    W_S[idx(0,0)][0] = 1; W_S[idx(1,1)][0] = 1
    
    # M2: (A10 + A11)B00
    U_S[idx(1,0)][1] = 1; U_S[idx(1,1)][1] = 1
    V_S[idx(0,0)][1] = 1
    W_S[idx(0,0)][1] = 1; W_S[idx(1,1)][1] = -1
    
    # M3: A00(B01 - B11)
    U_S[idx(0,0)][2] = 1
    V_S[idx(0,1)][2] = 1; V_S[idx(1,1)][2] = -1
    W_S[idx(0,1)][2] = 1; W_S[idx(1,1)][2] = 1
    
    # M4: A11(B10 - B00)
    U_S[idx(1,1)][3] = 1
    V_S[idx(1,0)][3] = 1; V_S[idx(0,0)][3] = -1
    W_S[idx(0,0)][3] = 1; W_S[idx(1,0)][3] = 1
    
    # M5: (A00 + A01)B11
    U_S[idx(0,0)][4] = 1; U_S[idx(0,1)][4] = 1
    V_S[idx(1,1)][4] = 1
    W_S[idx(0,0)][4] = -1; W_S[idx(0,1)][4] = 1
    
    # M6: (A10 - A00)(B00 + B01)
    U_S[idx(1,0)][5] = 1; U_S[idx(0,0)][5] = -1
    V_S[idx(0,0)][5] = 1; V_S[idx(0,1)][5] = 1
    W_S[idx(1,1)][5] = 1
    
    # M7: (A01 - A11)(B10 + B11)
    U_S[idx(0,1)][6] = 1; U_S[idx(1,1)][6] = -1
    V_S[idx(1,0)][6] = 1; V_S[idx(1,1)][6] = 1
    W_S[idx(0,0)][6] = 1

    # Schoolbook for 3x3
    # We iterate over all 27 terms
    # Term s corresponds to local i, j, k
    # U_L[l_i, l_j, s] = 1
    # V_L[l_j, l_k, s] = 1
    # W_L[l_k, l_i, s] = 1
    
    col = 0
    for k_s in range(7):
        for l_i in range(3):
            for l_j in range(3):
                for l_k in range(3):
                    # This is one term in the tensor product decomposition
                    # U contribution
                    for i in range(6):
                        for j in range(6):
                            b_i = i // 3
                            b_j = j // 3
                            li = i % 3
                            lj = j % 3
                            
                            u_s_val = U_S[idx(b_i, b_j)][k_s]
                            if u_s_val != 0:
                                # Check if local indices match schoolbook term
                                if li == l_i and lj == l_j:
                                    U[i * 6 + j][col] = u_s_val
                            
                    # V contribution
                    for j in range(6):
                        for k in range(6):
                            b_j = j // 3
                            b_k = k // 3
                            lj = j % 3
                            lk = k % 3
                            
                            v_s_val = V_S[idx(b_j, b_k)][k_s]
                            if v_s_val != 0:
                                if lj == l_j and lk == l_k:
                                    V[j * 6 + k][col] = v_s_val
                                    
                    # W contribution
                    for k in range(6):
                        for i in range(6):
                            b_k = k // 3
                            b_i = i // 3
                            lk = k % 3
                            li = i % 3
                            
                            w_s_val = W_S[idx(b_k, b_i)][k_s]
                            if w_s_val != 0:
                                if lk == l_k and li == l_i:
                                    W[k * 6 + i][col] = w_s_val
                                    
                    col += 1
                    if time.time() > deadline:
                        # Return partial result? No, must be complete.
                        # But 189 terms is very fast.
                        break
            if time.time() > deadline: break
        if time.time() > deadline: break
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
