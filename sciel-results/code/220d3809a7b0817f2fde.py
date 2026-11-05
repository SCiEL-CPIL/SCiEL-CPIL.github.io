"""Matrix multiplication 4x4x5 with Strassen(4x4x4) + schoolbook(4x4x1) decomposition.

Rank = 49 + 16 = 65
"""

def construct():
    n, m, p = 4, 4, 5
    rank = 65

    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    # --- Strassen 2x2x2 base case (rank 7) ---
    # Indices: A[i,j] i,j in {0,1}; B[j,k] j,k in {0,1}; C[i,k] i,k in {0,1}
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    s2_u = [[0]*7 for _ in range(4)]
    s2_v = [[0]*7 for _ in range(4)]
    s2_w = [[0]*7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    s2_u[0][0] = 1; s2_u[3][0] = 1
    s2_v[0][0] = 1; s2_v[3][0] = 1
    s2_w[0][0] = 1; s2_w[3][0] = 1

    # M2 = (A10+A11)B00 -> C10 += 1
    s2_u[2][1] = 1; s2_u[3][1] = 1
    s2_v[0][1] = 1
    s2_w[2][1] = 1

    # M3 = A00(B01-B11) -> C01 += 1
    s2_u[0][2] = 1
    s2_v[1][2] = 1; s2_v[3][2] = -1
    s2_w[1][2] = 1

    # M4 = A11(B10-B00) -> C10 += 1, C11 += 1
    s2_u[3][3] = 1
    s2_v[2][3] = 1; s2_v[0][3] = -1
    s2_w[2][3] = 1; s2_w[3][3] = 1

    # M5 = (A00+A01)B11 -> C01 += 1, C11 -= 1
    s2_u[0][4] = 1; s2_u[1][4] = 1
    s2_v[3][4] = 1
    s2_w[1][4] = 1; s2_w[3][4] = -1

    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    s2_u[2][5] = 1; s2_u[0][5] = -1
    s2_v[0][5] = 1; s2_v[1][5] = 1
    s2_w[3][5] = 1

    # M7 = (A01-A11)(B10+B11) -> C01 += 1, C11 += 1
    s2_u[1][6] = 1; s2_u[3][6] = -1
    s2_v[2][6] = 1; s2_v[3][6] = 1
    s2_w[1][6] = 1; s2_w[3][6] = 1

    # --- Build Strassen 4x4x4 (rank 49) via Kronecker product ---
    # For each pair (a,b) in {0..6}^2, column = a*7+b
    # U[i*4+j][col] = s2_u[i*2+a_idx][a] * s2_u[j*2+b_idx][b]
    # Wait - the Kronecker structure for recursive Strassen:
    # The 4x4x4 tensor = sum_{a,b} (s2_a ⊗ s2_b) where s2_a is the a-th rank-1 factor
    # U_4x4[i,j] = sum over the Kronecker: for column (a,b), U[i*4+j] = s2_u[i_hi*2 + a_local][a] * s2_u[i_lo*2 + b_local][b]
    # Actually the standard recursive Strassen Kronecker product:
    # If T_2 = sum_c u_c ⊗ v_c ⊗ w_c (rank 7), then T_4 = sum_{c1,c2} (u_{c1}⊗u_{c2}) ⊗ (v_{c1}⊗v_{c2}) ⊗ (w_{c1}⊗w_{c2})
    # where u_{c1}⊗u_{c2} is a vector of length 4x4=16 (flattened), etc.

    # Let me be more careful. For 2x2x2:
    # U is 4x7: row index = i*2+j (i=row of A, j=col of A)
    # V is 4x7: row index = j*2+k (j=row of B, k=col of B)  
    # W is 4x7: row index = k*2+i (k=row of C, i=col of C)
    # Wait, the task says W[k*n+i, r] for output C[i,k]... let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W row index = k*n + i where C[i,k] is the output element.
    # For 2x2x2: W row = k*2+i
    # For 4x4x5: W row = k*4+i (k in 0..4, i in 0..3)

    # For the Kronecker product of two 2x2x2 tensors:
    # A_4 = A_hi ⊗ A_lo (4x4), B_4 = B_hi ⊗ B_lo, C_4 = C_hi ⊗ C_lo
    # U_4[i_hi*4+j_hi, i_lo*4+j_lo] ... no wait.
    # A is 4x4: A[i,j] where i,j in 0..3. Split i = i_hi*2 + i_lo, j = j_hi*2 + j_lo.
    # U_4 row index = i*4 + j = (i_hi*2+i_lo)*4 + (j_hi*2+j_lo) = i_hi*8 + i_lo*4 + j_hi*2 + j_lo
    # The Kronecker: U_4[i,j] for column (c1,c2) = s2_u[i_hi*2+j_hi][c1] * s2_u[i_lo*2+j_lo][c2]
    # Similarly V_4[j,k] for column (c1,c2) = s2_v[j_hi*2+k_hi][c1] * s2_v[j_lo*2+k_lo][c2]
    # W_4[k,i] for column (c1,c2) = s2_w[k_hi*2+i_hi][c1] * s2_w[k_lo*2+i_lo][c2]

    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for i_hi in range(2):
                for j_hi in range(2):
                    for i_lo in range(2):
                        for j_lo in range(2):
                            i = i_hi * 2 + i_lo
                            j = j_hi * 2 + j_lo
                            u_idx = i * 4 + j
                            val_u = s2_u[i_hi * 2 + j_hi][c1] * s2_u[i_lo * 2 + j_lo][c2]
                            U[u_idx][col] = val_u

                            # V: row index = j*4 + k for 4x4 (but we only need first 4 cols of B)
                            # j in 0..3, k in 0..3
                            v_idx = j * 4 + k_hi  # placeholder, will compute properly below

            # Actually let me fill V and W in the same loop
            for i_hi in range(2):
                for j_hi in range(2):
                    for i_lo in range(2):
                        for j_lo in range(2):
                            for k_hi in range(2):
                                for k_lo in range(2):
                                    i = i_hi * 2 + i_lo
                                    j = j_hi * 2 + j_lo
                                    k = k_hi * 2 + k_lo

                                    u_idx = i * 4 + j
                                    U[u_idx][col] = s2_u[i_hi*2+j_hi][c1] * s2_u[i_lo*2+j_lo][c2]

                                    # V index for 4x4 part: j*4 + k (j in 0..3, k in 0..3)
                                    v_idx = j * 4 + k
                                    V[v_idx][col] = s2_v[j_hi*2+k_hi][c1] * s2_v[j_lo*2+k_lo][c2]

                                    # W index: k*4 + i (k in 0..3, i in 0..3) for 4x4x4 part
                                    w_idx = k * 4 + i
                                    W[w_idx][col] = s2_w[k_hi*2+i_hi][c1] * s2_w[k_lo*2+i_lo][c2]

    # --- Schoolbook for 4x4x1 (rank 16) ---
    # B has 5 columns; first 4 handled by Strassen, 5th (index 4) handled by schoolbook
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1  # B[j, 4]
            W[4*4+i][col] = 1  # C[i, 4] -> W row = 4*4 + i = 16+i
            col += 1

    return {"U": U, "V": V, "W": W}
