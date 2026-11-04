import numpy as np
import time

_DEADLINE = time.time() + 108

N, M, P = 4, 4, 7

def construct():
    t0 = time.time()

    # Precompute the standard Strassen 2x2x2 term table once
    # Each entry: (u_dict, v_dict, w_dict) where keys are (row,col) in 2x2
    # u_dict maps A positions, v_dict maps B positions, w_dict maps C positions
    strassen2 = []
    # M1 = (A00+A11)*(B00+B11) -> C00+=, C11+=
    strassen2.append((
        {(0,0):1.0, (1,1):1.0},
        {(0,0):1.0, (1,1):1.0},
        {(0,0):1.0, (1,1):1.0}
    ))
    # M2 = (A10+A11)*B00 -> C10+=, C11-=
    strassen2.append((
        {(1,0):1.0, (1,1):1.0},
        {(0,0):1.0},
        {(1,0):1.0, (1,1):-1.0}
    ))
    # M3 = A00*(B01-B11) -> C01+=, C11+=
    strassen2.append((
        {(0,0):1.0},
        {(0,1):1.0, (1,1):-1.0},
        {(0,1):1.0, (1,1):1.0}
    ))
    # M4 = A11*(B10-B00) -> C00+=, C10+=
    strassen2.append((
        {(1,1):1.0},
        {(1,0):1.0, (0,0):-1.0},
        {(0,0):1.0, (1,0):1.0}
    ))
    # M5 = (A00+A01)*B11 -> C00-=, C01+=
    strassen2.append((
        {(0,0):1.0, (0,1):1.0},
        {(1,1):1.0},
        {(0,0):-1.0, (0,1):1.0}
    ))
    # M6 = (A10-A00)*(B00+B01) -> C11+=
    strassen2.append((
        {(1,0):1.0, (0,0):-1.0},
        {(0,0):1.0, (0,1):1.0},
        {(1,1):1.0}
    ))
    # M7 = (A01-A11)*(B10+B11) -> C00+=
    strassen2.append((
        {(0,1):1.0, (1,1):-1.0},
        {(1,0):1.0, (1,1):1.0},
        {(0,0):1.0}
    ))

    # Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C) using block Strassen
    # This gives 49 terms
    rank1 = 49
    # Part 2: 4x4x3 (cols 4-6 of B, rows 0-3 of C)
    # Split into 4 independent 2x2x3 problems, each rank 11 -> total 44
    rank2 = 4 * 11
    total_rank = rank1 + rank2

    U = np.zeros((N*M, total_rank), dtype=float)
    V = np.zeros((M*P, total_rank), dtype=float)
    W = np.zeros((P*N, total_rank), dtype=float)

    # --- Part 1: 4x4x4 Strassen (rank 49) ---
    # 4x4 = 4 blocks of 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    # Each C block is computed via Strassen on the appropriate A and B blocks.
    # The 49 terms come from: for each of 7 Strassen "multiply" steps,
    # we need to form linear combinations of A-blocks and B-blocks,
    # and the result contributes to C-blocks.
    #
    # Actually, for 4x4x4, the standard approach is:
    # - Use the 7 Strassen products where each product is a 2x2x2 multiplication
    #   of (2x2 linear combos of A-blocks) x (2x2 linear combos of B-blocks)
    # - Each such product gives a 2x2 result that is a linear combo of C-blocks
    # - Total: 7 * 7 = 49 scalar multiplications
    #
    # Let me enumerate:
    # The 7 Strassen steps for 4x4:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    #
    # Where A00, A01, etc. are 2x2 blocks.
    # Each Mi is a 2x2 matrix.
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So for each Strassen step s (0..6), and each 2x2 sub-term t (0..6),
    # we have one scalar multiplication term.
    # Total: 7 * 7 = 49 terms.

    # For the 4x4x4 part, A indices are (i,j) with i,j in 0..3
    # A index = i*4 + j
    # B index = j*7 + k, but for cols 0-3: B index = j*7 + k where k in 0..3
    # C index = k*4 + i, for rows 0-3, cols 0-3

    for s in range(7):
        # Define which A-blocks and B-blocks are combined, and with what coefficients
        # s=0: M1: A-blocks {00:1, 11:1}, B-blocks {00:1, 11:1}
        # s=1: M2: A-blocks {10:1, 11:1}, B-blocks {00:1}
        # s=2: M3: A-blocks {00:1}, B-blocks {01:1, 11:-1}
        # s=3: M4: A-blocks {11:1}, B-blocks {10:1, 00:-1}
        # s=4: M5: A-blocks {00:1, 01:1}, B-blocks {11:1}
        # s=5: M6: A-blocks {10:1, 00:-1}, B-blocks {00:1, 01:1}
        # s=6: M7: A-blocks {01:1, 11:-1}, B-blocks {10:1, 11:1}

        a_block_coeffs = [
            {(0,0):1.0, (1,1):1.0},
            {(1,0):1.0, (1,1):1.0},
            {(0,0):1.0},
            {(1,1):1.0},
            {(0,0):1.0, (0,1):1.0},
            {(1,0):1.0, (0,0):-1.0},
            {(0,1):1.0, (1,1):-1.0}
        ][s]

        b_block_coeffs = [
            {(0,0):1.0, (1,1):1.0},
            {(0,0):1.0},
            {(0,1):1.0, (1,1):-1.0},
            {(1,0):1.0, (0,0):-1.0},
            {(1,1):1.0},
            {(0,0):1.0, (0,1):1.0},
            {(1,0):1.0, (1,1):1.0}
        ][s]

        # C block coefficients for this Strassen step
        # C00 = M1 + M4 - M5 + M7  -> s=0:+1, s=3:+1, s=4:-1, s=6:+1
        # C01 = M3 + M5            -> s=2:+1, s=4:+1
        # C10 = M2 + M4            -> s=1:+1, s=3:+1
        # C11 = M1 - M2 + M3 + M6  -> s=0:+1, s=1:-1, s=2:+1, s=5:+1
        c_block_coeffs = [
            {(0,0):1.0, (1,1):1.0},   # s=0 (M1)
            {(1,0):1.0, (1,1):-1.0},  # s=1 (M2)
            {(0,1):1.0, (1,1):1.0},   # s=2 (M3)
            {(0,0):1.0, (1,0):1.0},   # s=3 (M4)
            {(0,0):-1.0, (0,1):1.0},  # s=4 (M5)
            {(1,1):1.0},              # s=5 (M6)
            {(0,0):1.0}               # s=6 (M7)
        ][s]

        for t in range(7):
            r = s * 7 + t
            u_sub, v_sub, w_sub = strassen2[t]

            # Fill U for this term
            u_vec = np.zeros(N*M, dtype=float)
            for (li, lj), cu in u_sub.items():
                for (br, bc), ca in a_block_coeffs.items():
                    i = 2*br + li
                    j = 2*bc + lj
                    u_vec[i*M + j] += cu * ca
            U[:, r] = u_vec

            # Fill V for this term (cols 0-3 only)
            v_vec = np.zeros(M*P, dtype=float)
            for (li, lj), cv in v_sub.items():
                for (br, bc), cb in b_block_coeffs.items():
                    i = 2*br + li
                    j = 2*bc + lj  # col in 0..3
                    v_vec[i*P + j] += cv * cb
            V[:, r] = v_vec

            # Fill W for this term (rows 0-3, cols 0-3)
            w_vec = np.zeros(P*N, dtype=float)
            for (li, lj), cw in w_sub.items():
                for (br, bc), cc in c_block_coeffs.items():
                    i = 2*br + li
                    k = 2*bc + lj  # col in 0..3
                    w_vec[k*N + i] += cw * cc
            W[:, r] = w_vec

    # --- Part 2: 4x4x3 (cols 4-6 of B, rows 0-3 of C) ---
    # Split into 4 independent 2x2x3 problems:
    # A_00 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    # A_01 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    # A_10 (rows 2-3, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    # A_11 (rows 2-3, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    #
    # Each 2x2x3 has rank 11: 7 Strassen (for cols 4,5) + 4 schoolbook (for col 6)

    def add_2x2x3(a_rb, a_cb, b_rb, c_rb, base_rank):
        """Add 11 terms for a 2x2x3 multiplication.
        a_rb: A row block (0 or 1), a_cb: A col block (0 or 1)
        b_rb: B row block (0 or 1), c_rb: C row block (0 or 1)
        """
        # Strassen part for cols 4,5 (7 terms)
        for t in range(7):
            r = base_rank + t
            u_sub, v_sub, w_sub = strassen2[t]

            u_vec = np.zeros(N*M, dtype=float)
            for (li, lj), cu in u_sub.items():
                i = 2*a_rb + li
                j = 2*a_cb + lj
                u_vec[i*M + j] = cu
            U[:, r] = u_vec

            v_vec = np.zeros(M*P, dtype=float)
            for (li, lj), cv in v_sub.items():
                i = 2*b_rb + li
                j = 4 + lj  # col 4 or 5
                v_vec[i*P + j] = cv
            V[:, r] = v_vec

            w_vec = np.zeros(P*N, dtype=float)
            for (li, lj), cw in w_sub.items():
                i = 2*c_rb + li
                k = 4 + lj  # col 4 or 5
                w_vec[k*N + i] = cw
            W[:, r] = w_vec

        # Schoolbook part for col 6 (4 terms)
        schoolbook = [
            ((0,0), (0,0), 0),  # A[0,0]*B[0,6] -> C[0,6]
            ((0,1), (1,0), 0),  # A[0,1]*B[1,6] -> C[0,6]
            ((1,0), (0,0), 1),  # A[1,0]*B[0,6] -> C[1,6]
            ((1,1), (1,0), 1),  # A[1,1]*B[1,6] -> C[1,6]
        ]
        for idx, (a_pos, b_pos, c_row) in enumerate(schoolbook):
            r = base_rank + 7 + idx
            u_vec = np.zeros(N*M, dtype=float)
            i_a = 2*a_rb + a_pos[0]
            j_a = 2*a_cb + a_pos[1]
            u_vec[i_a*M + j_a] = 1.0
            U[:, r] = u_vec

            v_vec = np.zeros(M*P, dtype=float)
            i_b = 2*b_rb + b_pos[0]
            v_vec[i_b*P + 6] = 1.0
            V[:, r] = v_vec

            w_vec = np.zeros(P*N, dtype=float)
            i_c = 2*c_rb + c_row
            w_vec[6*N + i_c] = 1.0
            W[:, r] = w_vec

    add_2x2x3(0, 0, 0, 0, rank1)    # A00 * B_top -> C_top
    add_2x2x3(0, 1, 1, 0, rank1+11) # A01 * B_bot -> C_top
    add_2x2x3(1, 0, 0, 1, rank1+22) # A10 * B_top -> C_bot
    add_2x2x3(1, 1, 1, 1, rank1+33) # A11 * B_bot -> C_bot

    # Verify
    if time.time() < _DEADLINE:
        target = np.zeros((N*M, M*P, P*N), dtype=float)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    target[i*M+j, j*P+k, k*N+i] = 1.0

        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        max_err = np.max(np.abs(recon - target))

        if max_err > 1e-9:
            # Fallback to schoolbook
            rank_sb = N * M * P
            U_sb = np.zeros((N*M, rank_sb), dtype=float)
            V_sb = np.zeros((M*P, rank_sb), dtype=float)
            W_sb = np.zeros((P*N, rank_sb), dtype=float)
            r = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        U_sb[i*M+j, r] = 1.0
                        V_sb[j*P+k, r] = 1.0
                        W_sb[k*N+i, r] = 1.0
                        r += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
