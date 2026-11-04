    for t in range(5):
        base_col = t * 11
        
        # Global V indices for B rows 0, 1, 2 and cols 2t, 2t+1
        # V_idx[j][k_offset] = j * p + (2*t + k_offset)
        v_0_0 = 0 * p + 2*t
        v_0_1 = 0 * p + 2*t + 1
        v_1_0 = 1 * p + 2*t
        v_1_1 = 1 * p + 2*t + 1
        v_2_0 = 2 * p + 2*t
        v_2_1 = 2 * p + 2*t + 1
        
        # Global W indices for C rows 0, 1 and cols 2t, 2t+1
        # W_idx[k][i] = k * n + i
        w_0_0 = 2*t * n + 0
        w_0_1 = 2*t * n + 1
        w_1_0 = (2*t + 1) * n + 0
        w_1_1 = (2*t + 1) * n + 1
        
        # U indices are global and constant for all blocks (since A is fixed)
        # u_0_0 = 0, u_0_1 = 1, u_0_2 = 2
        # u_1_0 = 3, u_1_1 = 4, u_1_2 = 5
        
        # Strassen part (uses A cols 0,1 and B rows 0,1)
        # Term 1: M1
        U[0][base_col + 0] += 1 # A00
        U[4][base_col + 0] += 1 # A11
        V[v_0_0][base_col + 0] += 1 # B00
        V[v_1_1][base_col + 0] += 1 # B11
        W[w_0_0][base_col + 0] += 1 # C00
        W[w_1_1][base_col + 0] += 1 # C11

        # Term 2: M2
        U[3][base_col + 1] += 1 # A10
        U[4][base_col + 1] += 1 # A11
        V[v_1_1][base_col + 1] += 1 # B11
        W[w_1_0][base_col + 1] += 1 # C10
        W[w_1_1][base_col + 1] += 1 # C11

        # Term 3: M3
        U[0][base_col + 2] += 1 # A00
        V[v_0_1][base_col + 2] += 1 # B01
        V[v_1_1][base_col + 2] -= 1 # B11
        W[w_0_1][base_col + 2] += 1 # C01

        # Term 4: M4
        U[4][base_col + 3] += 1 # A11
        V[v_1_0][base_col + 3] += 1 # B10
        V[v_0_0][base_col + 3] -= 1 # B00
        W[w_0_0][base_col + 3] -= 1 # C00
        W[w_1_0][base_col + 3] += 1 # C10

        # Term 5: M5
        U[0][base_col + 4] += 1 # A00
        U[1][base_col + 4] += 1 # A01
        V[v_0_0][base_col + 4] += 1 # B00
        W[w_0_0][base_col + 4] += 1 # C00
        W[w_0_1][base_col + 4] += 1 # C01

        # Term 6: M6
        U[3][base_col + 5] += 1 # A10
        U[0][base_col + 5] -= 1 # A00
        V[v_0_0][base_col + 5] += 1 # B00
        V[v_0_1][base_col + 5] += 1 # B01
        W[w_1_0][base_col + 5] += 1 # C10

        # Term 7: M7
        U[1][base_col + 6] += 1 # A01
        U[4][base_col + 6] -= 1 # A11
        V[v_1_0][base_col + 6] += 1 # B10
        V[v_1_1][base_col + 6] += 1 # B11
        W[w_0_0][base_col + 6] -= 1 # C00
        W[w_1_1][base_col + 6] += 1 # C11

        # Linear part (uses A col 2 and B row 2)
        # C00 += A02 B20
        U[2][base_col + 7] += 1 # A02
        V[v_2_0][base_col + 7] += 1 # B20
        W[w_0_0][base_col + 7] += 1 # C00

        # C01 += A02 B21
        U[2][base_col + 8] += 1 # A02
        V[v_2_1][base_col + 8] += 1 # B21
        W[w_0_1][base_col + 8] += 1 # C01

        # C10 += A12 B20
        U[5][base_col + 9] += 1 # A12
        V[v_2_0][base_col + 9] += 1 # B20
        W[w_1_0][base_col + 9] += 1 # C10

        # C11 += A12 B21
        U[5][base_col + 10] += 1 # A12
        V[v_2_1][base_col + 10] += 1 # B21
        W[w_1_1][base_col + 10] += 1 # C11
