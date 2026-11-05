"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 10s safety margin
    n, m, p = DIMENSIONS

    # Strassen 2x2 factors (rank 7)
    # U_S[r, c]: row = i*2+j, col = c in 0..6
    # V_S[r, c]: row = j*2+k, col = c in 0..6
    # W_S[r, c]: row = k*2+i, col = c in 0..6
    #
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # Build U_S (4 x 7)
    U_S = [[0]*7 for _ in range(4)]
    # Row 0: A00, Row 1: A01, Row 2: A10, Row 3: A11
    U_S[0][0] = 1  # M1: A00
    U_S[0][2] = 1  # M3: A00
    U_S[0][4] = 1  # M5: A00
    U_S[0][5] = -1 # M6: -A00
    U_S[1][4] = 1  # M5: A01
    U_S[1][6] = 1  # M7: A01
    U_S[2][1] = 1  # M2: A10
    U_S[2][5] = 1  # M6: A10
    U_S[3][0] = 1  # M1: A11
    U_S[3][1] = 1  # M2: A11
    U_S[3][3] = 1  # M4: A11
    U_S[3][6] = -1 # M7: -A11

    # Build V_S (4 x 7)
    V_S = [[0]*7 for _ in range(4)]
    # Row 0: B00, Row 1: B01, Row 2: B10, Row 3: B11
    V_S[0][0] = 1  # M1: B00
    V_S[0][1] = 1  # M2: B00
    V_S[0][5] = 1  # M6: B00
    V_S[1][3] = -1 # M4: -B10... wait, M4 = A11(B01-B10) so V[1][3]=1, V[2][3]=-1
    # Let me redo:
    # M1: (A00+A11)(B00+B11) -> U: A00,A11; V: B00,B11
    # M2: (A10+A11)B00 -> U: A10,A11; V: B00
    # M3: A00(B10-B11) -> U: A00; V: B10,B11
    # M4: A11(B01-B10) -> U: A11; V: B01,B10
    # M5: (A00+A01)B11 -> U: A00,A01; V: B11
    # M6: (A10-A00)(B00+B10) -> U: A10,A00; V: B00,B10
    # M7: (A01-A11)(B01+B11) -> U: A01,A11; V: B01,B11

    V_S = [[0]*7 for _ in range(4)]
    # Row 0: B00
    V_S[0][0] = 1  # M1
    V_S[0][1] = 1  # M2
    V_S[0][5] = 1  # M6
    # Row 1: B01
    V_S[1][3] = 1  # M4
    V_S[1][6] = 1  # M7
    # Row 2: B10
    V_S[2][2] = 1  # M3
    V_S[2][3] = -1 # M4
    V_S[2][5] = 1  # M6
    # Row 3: B11
    V_S[3][0] = 1  # M1
    V_S[3][2] = -1 # M3
    V_S[3][4] = 1  # M5
    V_S[3][6] = 1  # M7

    # Build W_S (4 x 7)
    W_S = [[0]*7 for _ in range(4)]
    # Row 0: C00, Row 1: C01, Row 2: C10, Row 3: C11
    # C00 = M1 + M4 - M5 - M7
    W_S[0][0] = 1
    W_S[0][3] = 1
    W_S[0][4] = -1
    W_S[0][6] = -1
    # C01 = M3 + M5
    W_S[1][2] = 1
    W_S[1][4] = 1
    # C10 = M2 + M4
    W_S[2][1] = 1
    W_S[2][3] = 1
    # C11 = M1 - M2 + M3 + M7
    W_S[3][0] = 1
    W_S[3][1] = -1
    W_S[3][2] = 1
    W_S[3][6] = 1

    # Schoolbook 3x3 factors (rank 27)
    # U_3[i*3+j, i*9+j*3+k] = 1
    # V_3[j*3+k, i*9+j*3+k] = 1
    # W_3[i*3+k, i*9+j*3+k] = 1
    U_3 = [[0]*27 for _ in range(9)]
    V_3 = [[0]*27 for _ in range(9)]
    W_3 = [[0]*27 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*9 + j*3 + k
                U_3[i*3+j][c] = 1
                V_3[j*3+k][c] = 1
                W_3[i*3+k][c] = 1

    # Kronecker product: 6x6 = 3x3 (outer) ⊗ 2x2 (inner)
    # The 6x6 matrix is partitioned into 3x3 blocks of 2x2 submatrices.
    # A[3*i+2*a + 2*a' + b, 3*j+2*b+2*b'+c] ... 
    # Actually, let's think of it as: index in 6x6 = outer_index * 2 + inner_index
    # outer_index ranges 0..2 (3x3 grid), inner_index ranges 0..1 (2x2 block)
    # So 6x6 element (I, J) where I = 3*i + 2*a, J = 3*j + 2*b
    # corresponds to outer (i,j) and inner (a,b).
    #
    # The Kronecker product decomposition:
    # For each outer term (i,j,k) [3x3 schoolbook, 27 terms] and inner term (a,b,c) [2x2 Strassen, 7 terms]:
    # Total 27*7 = 189 terms.
    #
    # U_6[I*6+J, term] = U_3[i*3+j, outer_term] * U_2[a*2+b, inner_term]
    #   where I = 3*i + 2*a, J = 3*j + 2*b
    #   outer_term = i*9+j*3+k (0..26), inner_term = 0..6
    #   term = outer_term * 7 + inner_term
    #
    # V_6[J*6+K, term] = V_3[j*3+k, outer_term] * V_2[b*2+c, inner_term]
    #   where K = 3*k + 2*c
    #
    # W_6[K*6+I, term] = W_3[i*3+k, outer_term] * W_2[c*2+a, inner_term]
    #   Note: W index is k*n+i for n=6, so K*6+I where K is row of result (from B's row = k outer + c inner)
    #   Actually W[k*n+i, c]: k is the column index of A / row index of B in the sum, i is row of result.
    #   For 6x6: W[K*6+I, term] where K = 3*k+2*c, I = 3*i+2*a

    rank = 27 * 7
    U_6 = [[0]*rank for _ in range(36)]
    V_6 = [[0]*rank for _ in range(36)]
    W_6 = [[0]*rank for _ in range(36)]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                outer = i*9 + j*3 + k
                for a in range(2):
                    for b in range(2):
                        # U row: I*6+J = (3*i+2*a)*6 + (3*j+2*b)
                        I_u = 3*i + 2*a
                        J_u = 3*j + 2*b
                        u_row = I_u*6 + J_u
                        for c in range(2):
                            inner = b*2 + c  # V_2 row index: j*2+k -> here b*2+c
                            # For U: U_2[a*2+b, inner_term]
                            # Wait, U_2 rows are indexed by (a,b) where a is row of A-block, b is col of A-block
                            # U_2[a*2+b, c_inner]
                            # Let me re-index: inner term index = 0..6
                            for c_inner in range(7):
                                term = outer * 7 + c_inner
                                val = U_3[i*3+j][outer] * U_S[a*2+b][c_inner]
                                if val != 0:
                                    U_6[u_row][term] += val

    for j in range(3):
        for k in range(3):
            for i in range(3):
                outer = i*9 + j*3 + k
                for b in range(2):
                    for c in range(2):
                        # V row: J*6+K = (3*j+2*b)*6 + (3*k+2*c)
                        J_v = 3*j + 2*b
                        K_v = 3*k + 2*c
                        v_row = J_v*6 + K_v
                        for c_inner in range(7):
                            term = outer * 7 + c_inner
                            val = V_3[j*3+k][outer] * V_S[b*2+c][c_inner]
                            if val != 0:
                                V_6[v_row][term] += val

    for k in range(3):
        for i in range(3):
            for j in range(3):
                outer = i*9 + j*3 + k
                for c in range(2):
                    for a in range(2):
                        # W row: K*6+I = (3*k+2*c)*6 + (3*i+2*a)
                        K_w = 3*k + 2*c
                        I_w = 3*i + 2*a
                        w_row = K_w*6 + I_w
                        for c_inner in range(7):
                            term = outer * 7 + c_inner
                            val = W_3[i*3+k][outer] * W_S[c*2+a][c_inner]
                            if val != 0:
                                W_6[w_row][term] += val

    if time.time() > deadline:
        pass  # already constructed, just return

    return {"U": U_6, "V": V_6, "W": W_6}
# EVOLVE-BLOCK-END
