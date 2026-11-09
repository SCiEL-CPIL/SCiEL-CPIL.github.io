def _build_rank22_decomposition(n, m, p):
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def _add_strassen_block(a_idx, b_idx, c_idx, col_start):
        col = col_start
        # Strassen 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        U[a_idx['00']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[a_idx['10']][col] += 1; U[a_idx['11']][col] += 1
        V[b_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        W[c_idx['11']][col] += -1
        col += 1
        # M3 = A00(B01-B11)
        U[a_idx['00']][col] += 1
        V[b_idx['01']][col] += 1; V[b_idx['11']][col] += -1
        W[c_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[a_idx['11']][col] += 1
        V[b_idx['10']][col] += 1; V[b_idx['00']][col] += -1
        W[c_idx['00']][col] += 1
        W[c_idx['10']][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[a_idx['00']][col] += 1; U[a_idx['01']][col] += 1
        V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += -1
        W[c_idx['01']][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[a_idx['10']][col] += 1; U[a_idx['00']][col] += -1
        V[b_idx['00']][col] += 1; V[b_idx['01']][col] += 1
        W[c_idx['11']][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[a_idx['01']][col] += 1; U[a_idx['11']][col] += -1
        V[b_idx['10']][col] += 1; V[b_idx['11']][col] += 1
        W[c_idx['00']][col] += 1
        return col

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    a1 = {'00': 0, '01': 1, '10': 3, '11': 4}
    # B indices: 0(0,0), 1(0,1), 4(1,0), 5(1,1)
    b1 = {'00': 0, '01': 1, '10': 4, '11': 5}
    # C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # Note: C is p*n = 4*2 = 8. Index k*n + i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3.
    c1 = {'00': 0, '01': 1, '10': 2, '11': 3}
    
    # Wait, previous c1 was {'00': 0, '01': 2, '10': 1, '11': 3}.
    # Let's check standard Strassen mapping.
    # C00 = M1 - M4 + M5 - M7 ? No.
    # Standard:
    # C00 = M1 - M4 + M5 - M7 ?
    # Let's stick to the verified Strassen implementation in the current code if it worked for 2x2.
    # The current code's `_add_strassen_block` is a standard Strassen.
    # The mapping of indices to C positions depends on the order of 00, 01, 10, 11.
    # In `_add_strassen_block`:
    # M1 contributes to C00 and C11.
    # M2 contributes to C10 and C11.
    # M3 contributes to C01 and C11.
    # M4 contributes to C00 and C10.
    # M5 contributes to C00 and C01.
    # M6 contributes to C11.
    # M7 contributes to C00.
    
    # Let's trace M1: W[c_idx['00']] += 1, W[c_idx['11']] += 1.
    # M2: W[c_idx['10']] += 1, W[c_idx['11']] += -1.
    # M3: W[c_idx['01']] += 1, W[c_idx['11']] += 1.
    # M4: W[c_idx['00']] += 1, W[c_idx['10']] += 1.
    # M5: W[c_idx['00']] += -1, W[c_idx['01']] += 1.
    # M6: W[c_idx['11']] += 1.
    # M7: W[c_idx['00']] += 1.
    
    # Sum for C00: M1 + M4 - M5 + M7.
    # Sum for C01: M3 + M5.
    # Sum for C10: M2 + M4.
    # Sum for C11: M1 - M2 + M3 + M6.
    
    # This is a valid Strassen decomposition.
    
    col = _add_strassen_block(a1, b1, c1, 0)

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: same as a1
    # B indices: 2(0,2), 3(0,3), 6(1,2), 7(1,3)
    b2 = {'00': 2, '01': 3, '10': 6, '11': 7}
    # C indices for C[0:2, 2:4]
    # C[0,2] = 2*2 + 0 = 4? No. Index k*n + i.
    # k is col (0..3), i is row (0..1).
    # C[0,2]: k=2, i=0 -> 2*2 + 0 = 4.
    # C[0,3]: k=3, i=0 -> 3*2 + 0 = 6.
    # C[1,2]: k=2, i=1 -> 2*2 + 1 = 5.
    # C[1,3]: k=3, i=1 -> 3*2 + 1 = 7.
    # So c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    # Wait, 00 is C[0,2], 01 is C[0,3], 10 is C[1,2], 11 is C[1,3].
    c2 = {'00': 4, '01': 6, '10': 5, '11': 7}
    
    col = _add_strassen_block(a1, b2, c2, col)

    # Block 3: A[:, 2] * B[2, :] -> C[:, :]
    # A indices: 2(0,2), 5(1,2)
    # B indices: 8(2,0), 9(2,1), 10(2,2), 11(2,3)
    # C indices: all 8.
    # We need 8 terms: A[0,2]B[2,k] and A[1,2]B[2,k].
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    return U, V, W
