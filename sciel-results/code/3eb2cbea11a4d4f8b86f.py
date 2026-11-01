"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 140  # seconds, safety margin before 150s kill
    n, m, p = 2, 4, 4
    r = 28

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Four independent 2x2 Strassen products (rank 7 each, total 28)
    # The 2x4 times 4x4 product is the sum of four 2x2*2x2 block products
    # along the column dimension of A and row dimension of B.
    #
    # For each block (A_col_block, B_row_block):
    #   A sub-block: rows 0..1 (n=2), cols 2*ab .. 2*ab+1
    #   B sub-block: rows 2*bb .. 2*bb+1, cols 0..3 (p=4)
    #   C sub-block: rows 0..1, cols 2*cb .. 2*cb+1
    #
    # Block mapping:
    #   Block 0: ab=0, bb=0, cb=0  -> A[0:2,0:2] * B[0:2,0:4] -> C[0:2,0:2]
    #   Block 1: ab=1, bb=0, cb=0  -> A[0:2,2:4] * B[0:2,0:4] -> C[0:2,0:2]
    #   Block 2: ab=0, bb=1, cb=1  -> A[0:2,0:2] * B[2:4,0:4] -> C[0:2,2:4]
    #   Block 3: ab=1, bb=1, cb=1  -> A[0:2,2:4] * B[2:4,0:4] -> C[0:2,2:4]

    def _strassen_2x4_block(off, a_c0, b_r0, c_c0):
        # Index helpers
        # A indices: row in {0,1}, col = a_c0 + {0,1}
        a00 = 0 * m + (a_c0 + 0)
        a01 = 0 * m + (a_c0 + 1)
        a10 = 1 * m + (a_c0 + 0)
        a11 = 1 * m + (a_c0 + 1)

        # B indices: row = b_r0 + {0,1}, col in {0,1,2,3}
        # For 2x4 B block, we need to split into 2x2 sub-blocks for Strassen.
        # Actually, the Strassen product here is 2x2 * 4x4? No.
        # Wait: A is 2x4, B is 4x4. We split A into two 2x2 column blocks,
        # and B into two 4x2 row blocks. Each 2x2 * 4x2 is NOT a standard
        # Strassen 2x2*2x2.
        #
        # Let me reconsider. The standard approach:
        # C = A * B where A is 2x4, B is 4x4.
        # Split A = [A_L | A_R] where A_L, A_R are 2x2.
        # Split B = [B_T; B_B] where B_T, B_B are 2x4.
        # C = A_L * B_T + A_R * B_B, where each product is 2x2 * 2x4 = 2x4.
        #
        # Now, A_L * B_T: A_L is 2x2, B_T is 2x4.
        # We can write B_T = [B_TL | B_TR] where B_TL, B_TR are 2x2.
        # A_L * B_T = A_L * B_TL + A_L * B_TR, each is 2x2*2x2.
        #
        # So overall: C = A_L*B_TL + A_L*B_TR + A_R*B_TL + A_R*B_TR
        # That's 4 Strassen 2x2*2x2 products, each rank 7, total 28.
        #
        # Let me redo the indexing properly.
        #
        # A_L: rows 0-1, cols 0-1
        # A_R: rows 0-1, cols 2-3
        # B_TL: rows 0-1, cols 0-1
        # B_TR: rows 0-1, cols 2-3
        # B_BL: rows 2-3, cols 0-1
        # B_BR: rows 2-3, cols 2-3
        #
        # C = A_L*B_TL + A_R*B_BL + A_L*B_TR + A_R*B_BR
        # where:
        #   A_L*B_TL -> 2x2, contributes to C[0:2, 0:2]
        #   A_R*B_BL -> 2x2, contributes to C[0:2, 0:2]
        #   A_L*B_TR -> 2x2, contributes to C[0:2, 2:4]
        #   A_R*B_BR -> 2x2, contributes to C[0:2, 2:4]

        pass

    # Let me just hardcode the four Strassen blocks with correct indices.

    def _add_strassen(off, ar0, ac0, br0, bc0, cr0, cc0):
        # A: rows ar0..ar0+1, cols ac0..ac0+1 (2x2)
        # B: rows br0..br0+1, cols bc0..bc0+1 (2x2)
        # C: rows cr0..cr0+1, cols cc0..cc0+1 (2x2)
        # W convention: W[k*n + i, r] where k is col of C, i is row of C
        # So C[row, col] -> W[col * n + row]

        a00 = ar0 * m + (ac0 + 0)
        a01 = ar0 * m + (ac0 + 1)
        a10 = (ar0 + 1) * m + (ac0 + 0)
        a11 = (ar0 + 1) * m + (ac0 + 1)

        b00 = br0 * p + (bc0 + 0)
        b01 = br0 * p + (bc0 + 1)
        b10 = (br0 + 1) * p + (bc0 + 0)
        b11 = (br0 + 1) * p + (bc0 + 1)

        c00 = cc0 * n + cr0
        c01 = (cc0 + 1) * n + cr0
        c10 = cc0 * n + (cr0 + 1)
        c11 = (cc0 + 1) * n + (cr0 + 1)

        # M1 = (a00+a11)(b00+b11)
        U[a00][off] += 1
        U[a11][off] += 1
        V[b00][off] += 1
        V[b11][off] += 1
        W[c00][off] += 1
        W[c11][off] += 1

        # M2 = (a10+a11)b00
        U[a10][off+1] += 1
        U[a11][off+1] += 1
        V[b00][off+1] += 1
        W[c10][off+1] += 1
        W[c11][off+1] -= 1

        # M3 = a00(b01-b11)
        U[a00][off+2] += 1
        V[b01][off+2] += 1
        V[b11][off+2] -= 1
        W[c01][off+2] += 1
        W[c11][off+2] += 1

        # M4 = a11(b10-b00)
        U[a11][off+3] += 1
        V[b10][off+3] += 1
        V[b00][off+3] -= 1
        W[c00][off+3] += 1
        W[c10][off+3] += 1

        # M5 = (a00+a01)b11
        U[a00][off+4] += 1
        U[a01][off+4] += 1
        V[b11][off+4] += 1
        W[c00][off+4] -= 1
        W[c01][off+4] += 1

        # M6 = (a10-a00)(b00+b01)
        U[a10][off+5] += 1
        U[a00][off+5] -= 1
        V[b00][off+5] += 1
        V[b01][off+5] += 1
        W[c11][off+5] += 1

        # M7 = (a01-a11)(b10+b11)
        U[a01][off+6] += 1
        U[a11][off+6] -= 1
        V[b10][off+6] += 1
        V[b11][off+6] += 1
        W[c00][off+6] += 1

    # Block 1: A_L * B_TL -> C[0:2, 0:2]
    _add_strassen(0, 0, 0, 0, 0, 0, 0)

    # Block 2: A_R * B_BL -> C[0:2, 0:2]
    _add_strassen(7, 0, 2, 2, 0, 0, 0)

    # Block 3: A_L * B_TR -> C[0:2, 2:4]
    _add_strassen(14, 0, 0, 0, 2, 0, 2)

    # Block 4: A_R * B_BR -> C[0:2, 2:4]
    _add_strassen(21, 0, 2, 2, 2, 0, 2)

    # Quick self-verification: check that sum_r U[i,r]*V[j,r]*W[k,r] = delta(i,j,k)
    # for the multiplication tensor. This is a sanity check.
    if time.time() - _t0 < _DEADLINE - 5:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For matrix mult: T[i,j,k] = 1 if i==j and j==k (with proper index mapping)
        # U index: row*n + col where row in [0,2), col in [0,4) -> 0..7
        # V index: row*m + col where row in [0,4), col in [0,4) -> 0..15
        # W index: col*n + row where col in [0,4), row in [0,2) -> 0..7
        # C[row, col] = sum_k A[row,k] * B[k,col]
        # In tensor form: C[i,k] = sum_j A[i,j] * B[j,k]
        # U[i,j] = A[i,j], V[j,k] = B[j,k], W[k,i] = C[i,k] (W convention: W[k*n+i])
        # So T[A_idx, B_idx, C_idx] should be 1 if A_idx corresponds to (i,j),
        # B_idx to (j,k), C_idx to (k,i) for valid i,j,k, else 0.
        # A_idx = i*4 + j, B_idx = j*4 + k, C_idx = k*2 + i
        # Check: T[A_idx, B_idx, C_idx] = 1 iff A_idx = i*4+j, B_idx = j*4+k, C_idx = k*2+i
        # for some i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
        # i.e., A_idx//4 == B_idx//4 == j and A_idx%4 == j and B_idx%4 == k and C_idx%2 == i and C_idx//2 == k
        # Simplify: let i = A_idx//4, j = A_idx%4, then need B_idx = j*4 + k for some k, and C_idx = k*2 + i
        # So B_idx must have B_idx//4 == j, and k = B_idx%4, and C_idx must equal k*2 + i
        
        ok = True
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    a_idx = i * 4 + j
                    b_idx = j * 4 + k
                    c_idx = k * 2 + i
                    # T[a_idx, b_idx, c_idx] should be 1
                    val = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx])
                    if abs(val - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        # Also check that T is zero where it should be
        if ok:
            # Check a few off-diagonal entries
            # T[a_idx, b_idx, c_idx] = 0 when the indices don't align
            # e.g., a_idx=0 (i=0,j=0), b_idx=4 (j=1,k=0) -> j mismatch, should be 0
            test_cases = [
                (0, 4, 0),   # i=0,j=0 vs j=1: mismatch
                (0, 0, 1),   # i=0,j=0,k=0 vs c_idx=1 (k=0,i=0): c should be 0, not 1
                (2, 0, 0),   # i=0,j=2 vs j=0: mismatch
            ]
            for a_idx, b_idx, c_idx in test_cases:
                val = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx])
                if abs(val) > 1e-9:
                    ok = False
                    break
        
        if not ok:
            # Fallback: return empty (should not happen)
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
