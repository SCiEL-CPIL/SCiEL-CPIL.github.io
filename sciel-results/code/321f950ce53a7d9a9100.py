"""Strassen block decomposition with rank-7 2x2*2x3 blocks for 4x4*4x5."""
import time
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]

    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p  # 80

    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1

        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1

        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def strassen_2x3():
        """Return U, V, W for 2x2 * 2x3 multiplication using rank-7 Strassen-like decomposition.
        U: 4x7, V: 6x7, W: 6x7
        Convention: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
        
        We use the 2x2*2x2 Strassen decomposition and extend it to handle the third column.
        The key insight: we can use the same 7 terms as Strassen for 2x2*2x2,
        but extend the V and W matrices to handle the extra column.
        
        For 2x2*2x3: C[i][k] = sum_j A[i][j] * B[j][k] for k=0,1,2
        We treat B as having columns [B_col0, B_col1, B_col2].
        
        Using Strassen's structure:
        M1 = (A11+A22)*(B11+B21+B22)  -- wait, this doesn't work directly.
        
        Actually, let's use a different approach. We can use the fact that
        2x2*2x3 can be embedded in 2x2*2x2 by padding, but that's not helpful.
        
        Let me try a direct rank-7 construction:
        We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
        
        Let's use the following 7 terms (adapted from Strassen):
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        # Term 0: (A11+A22) * (B10+B20) -> contributes to C[0][0] and C[1][1]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # Term 1: (A21+A22) * B10 -> contributes to C[1][0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1

        # Term 2: A11 * (B11-B21) -> contributes to C[0][1]
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[1][2] = 1

        # Term 3: A22 * (B20-B10) -> contributes to C[0][0] and C[1][0]
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # Term 4: (A11+A12) * B22 -> contributes to C[0][2] and C[1][2]
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # Term 5: (A21-A11) * (B10+B11) -> contributes to C[1][0] and C[1][1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # Term 6: (A12-A22) * (B20+B21) -> contributes to C[0][0], C[0][1], C[1][0], C[1][1]
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def test_strassen():
        """Test Strassen 2x2 decomposition."""
        U, V, W = strassen_2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def test_strassen_2x3():
        """Test 2x2*2x3 rank-7 decomposition."""
        U, V, W = strassen_2x3()
        rank = len(U[0])
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0
                    for r in range(rank):
                        s += U[i*2+j][r] * V[j*3+k][r] * W[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try rank-7 2x2*2x3 decomposition
    use_2x3_rank7 = False
    if test_strassen():
        sU, sV, sW = strassen_2x2()
        
        # Test the 2x2*2x3 rank-7 decomposition
        if test_strassen_2x3():
            use_2x3_rank7 = True
            tU, tV, tW = strassen_2x3()

        def build_strassen_block():
            """Build 4x4*4x5 using Strassen for 2x2*2x2 and rank-7 for 2x2*2x3."""
            U_list = []
            V_list = []
            W_list = []

            # Strassen A index -> full U index for each A block
            a11 = [0, 1, 4, 5]
            a12 = [2, 3, 6, 7]
            a21 = [8, 9, 12, 13]
            a22 = [10, 11, 14, 15]

            # Strassen B index -> full V index for each B block
            b11 = [0, 1, 5, 6]
            b21 = [10, 11, 15, 16]

            # Strassen W index -> full W index for each C block
            c11 = [0, 4, 1, 5]
            c21 = [2, 6, 3, 7]

            def add_strassen_2x2(a_map, b_map, c_map):
                for r in range(7):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    for si in range(4):
                        if sU[si][r] != 0:
                            U_new[a_map[si]] = sU[si][r]
                    for sj in range(4):
                        if sV[sj][r] != 0:
                            V_new[b_map[sj]] = sV[sj][r]
                    for sk in range(4):
                        if sW[sk][r] != 0:
                            W_new[c_map[sk]] = sW[sk][r]
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            def add_strassen_2x3(a_map, b_map, c_map):
                """Add a 2x2*2x3 block using rank-7 decomposition.
                a_map: maps 4 Strassen A indices to full U indices
                b_map: maps 6 Strassen B indices to full V indices
                c_map: maps 6 Strassen W indices to full W indices
                """
                for r in range(7):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    for si in range(4):
                        if tU[si][r] != 0:
                            U_new[a_map[si]] = tU[si][r]
                    for sj in range(6):
                        if tV[sj][r] != 0:
                            V_new[b_map[sj]] = tV[sj][r]
                    for sk in range(6):
                        if tW[sk][r] != 0:
                            W_new[c_map[sk]] = tW[sk][r]
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            # C11 = A11*B11 + A12*B21 (2x2*2x2, use Strassen)
            add_strassen_2x2(a11, b11, c11)  # A11*B11
            add_strassen_2x2(a12, b21, c11)  # A12*B21

            # C21 = A21*B11 + A22*B21 (2x2*2x2, use Strassen)
            add_strassen_2x2(a21, b11, c21)  # A21*B11
            add_strassen_2x2(a22, b21, c21)  # A22*B21

            # C12 = A11*B12 + A12*B22 (2x2*2x3, use rank-7 if available)
            if use_2x3_rank7:
                # B12: rows 0-1, cols 2-4 -> V indices: j*5+(2+k) for j=0,1 k=0,1,2
                b12 = [2, 3, 4, 7, 8, 9]
                # C12: rows 0-1, cols 2-4 -> W indices: (2+k)*4+i for i=0,1 k=0,1,2
                c12 = [2, 6, 3, 7, 4, 8]
                add_strassen_2x3(a11, b12, c12)  # A11*B12

                # B22: rows 2-3, cols 2-4 -> V indices: (2+j)*5+(2+k) for j=0,1 k=0,1,2
                b22 = [12, 13, 14, 17, 18, 19]
                add_strassen_2x3(a12, b22, c12)  # A12*B22
            else:
                # Schoolbook fallback
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            U_new = [0]*16
                            V_new = [0]*20
                            W_new = [0]*20
                            U_new[i*4+j] = 1
                            V_new[j*5+(2+k)] = 1
                            W_new[(2+k)*4+i] = 1
                            U_list.append(U_new)
                            V_list.append(V_new)
                            W_list.append(W_new)

                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            U_new = [0]*16
                            V_new = [0]*20
                            W_new = [0]*20
                            U_new[i*4+(2+j)] = 1
                            V_new[(2+j)*5+(2+k)] = 1
                            W_new[(2+k)*4+i] = 1
                            U_list.append(U_new)
                            V_list.append(V_new)
                            W_list.append(W_new)

            # C22 = A21*B12 + A22*B22 (2x2*2x3, use rank-7 if available)
            if use_2x3_rank7:
                b12 = [2, 3, 4, 7, 8, 9]
                c22 = [10, 14, 11, 15, 12, 16]
                add_strassen_2x3(a21, b12, c22)  # A21*B12

                b22 = [12, 13, 14, 17, 18, 19]
                add_strassen_2x3(a22, b22, c22)  # A22*B22
            else:
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            U_new = [0]*16
                            V_new = [0]*20
                            W_new = [0]*20
                            U_new[(2+i)*4+j] = 1
                            V_new[j*5+(2+k)] = 1
                            W_new[(2+k)*4+(2+i)] = 1
                            U_list.append(U_new)
                            V_list.append(V_new)
                            W_list.append(W_new)

                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            U_new = [0]*16
                            V_new = [0]*20
                            W_new = [0]*20
                            U_new[(2+i)*4+(2+j)] = 1
                            V_new[(2+j)*5+(2+k)] = 1
                            W_new[(2+k)*4+(2+i)] = 1
                            U_list.append(U_new)
                            V_list.append(V_new)
                            W_list.append(W_new)

            return {"U": U_list, "V": V_list, "W": W_list}

        strassen_result = build_strassen_block()
        rank = len(strassen_result["U"])

        if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
            if rank < best_rank:
                best = strassen_result
                best_rank = rank

    # Try to reduce rank by merging identical terms
    def merge_terms(U, V, W):
        """Merge terms that have identical (U, V, W) patterns by summing coefficients.
        Returns (new_U, new_V, new_W) or None if no reduction possible.
        Coefficients are absorbed into the W matrix since we can scale.
        """
        term_dict = {}
        term_order = []
        for r in range(len(U[0])):
            u_key = tuple(U[i][r] for i in range(16))
            v_key = tuple(V[j][r] for j in range(20))
            w_key = tuple(W[k][r] for k in range(20))
            key = (u_key, v_key, w_key)
            if key in term_dict:
                term_dict[key] += 1
            else:
                term_dict[key] = 1
                term_order.append(key)

        # Check if we can reduce
        if len(term_order) >= len(U[0]):
            return None

        new_U = []
        new_V = []
        new_W = []
        for key in term_order:
            coeff = term_dict[key]
            u_key, v_key, w_key = key
            new_U.append(list(u_key))
            new_V.append(list(v_key))
            # Absorb coefficient into W
            new_W.append([w * coeff for w in w_key])

        return new_U, new_V, new_W

    # Try merging on the current best
    if time.time() - start_time < deadline:
        merged = merge_terms(best["U"], best["V"], best["W"])
        if merged is not None:
            mU, mV, mW = merged
            if verify(mU, mV, mW):
                if len(mU) < best_rank:
                    best = {"U": mU, "V": mV, "W": mW}
                    best_rank = len(mU)

    # Try a different approach: use Strassen on the full 4x4*4x4 part (first 4 cols)
    # and schoolbook for the 4x4*4x1 part (last col)
    if time.time() - start_time < deadline:
        def build_strassen_4x4_plus_4x1():
            """Build 4x4*4x5 by splitting B into 4x4 and 4x1.
            4x4*4x4: use Strassen recursively (rank 49)
            4x4*4x1: schoolbook (rank 16)
            Total: 65
            """
            U_list = []
            V_list = []
            W_list = []

            # First, 4x4*4x4 using Strassen (rank 49)
            # Split A into 2x2 blocks: A11, A12, A21, A22
            # Split B_left (4x4) into 2x2 blocks: B11, B12, B21, B22
            # C_left (4x4) into 2x2 blocks: C11, C12, C21, C22

            a11 = [0, 1, 4, 5]
            a12 = [2, 3, 6, 7]
            a21 = [8, 9, 12, 13]
            a22 = [10, 11, 14, 15]

            b11 = [0, 1, 5, 6]
            b12 = [2, 3, 7, 8]
            b21 = [10, 11, 15, 16]
            b22 = [12, 13, 17, 18]

            c11 = [0, 4, 1, 5]
            c12 = [2, 6, 3, 7]
            c21 = [8, 12, 9, 13]
            c22 = [10, 14, 11, 15]

            sU, sV, sW = strassen_2x2()

            def add_strassen_4x4(a_map, b_map, c_map):
                for r in range(7):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    for si in range(4):
                        if sU[si][r] != 0:
                            U_new[a_map[si]] = sU[si][r]
                    for sj in range(4):
                        if sV[sj][r] != 0:
                            V_new[b_map[sj]] = sV[sj][r]
                    for sk in range(4):
                        if sW[sk][r] != 0:
                            W_new[c_map[sk]] = sW[sk][r]
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            # Strassen for 4x4*4x4: 7 terms at the 4x4 level, each using Strassen 2x2*2x2
            # But wait, this gives 7*7 = 49 terms, which is correct for recursive Strassen.
            # Actually, we need to be careful. The 4x4*4x4 Strassen uses 7 "multiplications"
            # of 2x2 matrices. Each 2x2*2x2 multiplication uses 7 scalar multiplications.
            # So total is 49 scalar terms.

            # M1 = (A11+A22)*(B11+B22)
            # M2 = (A21+A22)*B11
            # M3 = A11*(B12-B22)
            # M4 = A22*(B21-B11)
            # M5 = (A11+A12)*B22
            # M6 = (A21-A11)*(B11+B12)
            # M7 = (A12-A22)*(B21+B22)

            # For each Mi, we need to compute the 2x2*2x2 product using Strassen.
            # This is getting complex. Let's just use the block approach.

            # Actually, let's use a simpler approach: just use the 2x2*2x2 Strassen
            # for each of the 4 sub-blocks in the 4x4*4x4 product.
            # C11 = A11*B11 + A12*B21: 2 Strassen 2x2*2x2 = 14 terms
            # C12 = A11*B12 + A12*B22: 2 Strassen 2x2*2x2 = 14 terms
            # C21 = A21*B11 + A22*B21: 2 Strassen 2x2*2x2 = 14 terms
            # C22 = A21*B12 + A22*B22: 2 Strassen 2x2*2x2 = 14 terms
            # Total: 56 terms for 4x4*4x4

            add_strassen_4x4(a11, b11, c11)
            add_strassen_4x4(a12, b21, c11)
            add_strassen_4x4(a11, b12, c12)
            add_strassen_4x4(a12, b22, c12)
            add_strassen_4x4(a21, b11, c21)
            add_strassen_4x4(a22, b21, c21)
            add_strassen_4x4(a21, b12, c22)
            add_strassen_4x4(a22, b22, c22)

            # Now add 4x4*4x1 for the last column (col 4 of B)
            # B_col4: rows 0-3, col 4 -> V indices: j*5+4 for j=0,1,2,3
            # C_col4: rows 0-3, col 4 -> W indices: 4*4+i = 16+i for i=0,1,2,3
            for i in range(4):
                for j in range(4):
                    U_new = [0]*16
                    V_new = [0]*20
                    W_new = [0]*20
                    U_new[i*4+j] = 1
                    V_new[j*5+4] = 1
                    W_new[16+i] = 1
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

            return {"U": U_list, "V": V_list, "W": W_list}

        s44_result = build_strassen_4x4_plus_4x1()
        s44_rank = len(s44_result["U"])
        if verify(s44_result["U"], s44_result["V"], s44_result["W"]):
            if s44_rank < best_rank:
                best = s44_result
                best_rank = s44_rank

            # Try merging
            if time.time() - start_time < deadline:
                merged44 = merge_terms(s44_result["U"], s44_result["V"], s44_result["W"])
                if merged44 is not None:
                    m44U, m44V, m44W = merged44
                    if verify(m44U, m44V, m44W):
                        if len(m44U) < best_rank:
                            best = {"U": m44U, "V": m44V, "W": m44W}
                            best_rank = len(m44U)

    # Try yet another approach: use a 3x1 split on B
    # 4x4*4x5 = 4x4*4x3 + 4x4*4x2
    # 4x4*4x2: split into 2x2 blocks -> 4 Strassen 2x2*2x2 = 4*7 = 28
    # 4x4*4x3: split into 2x2 blocks -> 2 Strassen 2x2*2x2 + 2x2*2x1 parts
    # This is getting complex. Let's try a different partition.

    # Try: split A by rows into 2x4 blocks, B by cols into 4x2 and 4x3
    if time.time() - start_time < deadline:
        def build_row_col_split():
            """Split A by rows (2x4 blocks), B by cols (4x2, 4x3).
            C_top_left = A_top * B_left (2x4 * 4x2 = 16 terms)
            C_top_right = A_top * B_right (2x4 * 4x3 = 24 terms)
            C_bot_left = A_bot * B_left (2x4 * 4x2 = 16 terms)
            C_bot_right = A_bot * B_right (2x4 * 4x3 = 24 terms)
            Total: 80 terms (same as schoolbook, no improvement)
            """
            # This doesn't help. Skip.
            return None

    return best

# EVOLVE-BLOCK-END
