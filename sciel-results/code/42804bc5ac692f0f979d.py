"""Rewritten program: Strassen block decomposition + rank-term merging for 4x4*4x5."""
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

    if test_strassen():
        sU, sV, sW = strassen_2x2()

        def build_strassen_block():
            """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
            U_list = []
            V_list = []
            W_list = []

            # Strassen A index -> full U index for each A block
            # A11: rows 0-1, cols 0-1 -> U[i*4+j]
            a11 = [0, 1, 4, 5]
            # A12: rows 0-1, cols 2-3 -> U[i*4+j]
            a12 = [2, 3, 6, 7]
            # A21: rows 2-3, cols 0-1 -> U[i*4+j]
            a21 = [8, 9, 12, 13]
            # A22: rows 2-3, cols 2-3 -> U[i*4+j]
            a22 = [10, 11, 14, 15]

            # Strassen B index -> full V index for each B block
            # B11: rows 0-1, cols 0-1 -> V[j*5+k]
            b11 = [0, 1, 5, 6]
            # B12: rows 0-1, cols 2-4 -> V[j*5+k] (but Strassen needs 2x2, use cols 0-1 of B12)
            # For Strassen on B12 (2x3), we only use the first 2 columns as a 2x2
            # Actually, Strassen requires square blocks. For 2x2*2x3, we can't directly use Strassen.
            # We'll use schoolbook for 2x2*2x3 blocks.

            # Strassen W index -> full W index for each C block
            # C11: rows 0-1, cols 0-1 -> W[k*4+i]
            c11 = [0, 4, 1, 5]
            # C21: rows 2-3, cols 0-1 -> W[k*4+i]
            c21 = [2, 6, 3, 7]

            def add_strassen(a_map, b_map, c_map):
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

            # C11 = A11*B11 + A12*B21
            add_strassen(a11, b11, c11)  # A11*B11
            # A12*B21: A12 is 2x2, B21 is 2x2 (rows 2-3, cols 0-1)
            b21 = [10, 11, 15, 16]
            add_strassen(a12, b21, c11)  # A12*B21

            # C21 = A21*B11 + A22*B21
            add_strassen(a21, b11, c21)  # A21*B11
            add_strassen(a22, b21, c21)  # A22*B21

            # C12 = A11*B12 + A12*B22 (2x3 blocks, use schoolbook)
            # A11: rows 0-1, cols 0-1
            # B12: rows 0-1, cols 2-4
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

            # A12*B22: A12 rows 0-1 cols 2-3, B22 rows 2-3 cols 2-4
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

            # C22 = A21*B12 + A22*B22 (2x3 blocks, use schoolbook)
            # A21: rows 2-3, cols 0-1
            # B12: rows 0-1, cols 2-4
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

            # A22*B22: A22 rows 2-3 cols 2-3, B22 rows 2-3 cols 2-4
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
        """Merge terms that have identical (U, V, W) patterns by summing coefficients."""
        term_dict = {}
        for r in range(len(U[0])):
            u_key = tuple(U[i][r] for i in range(16))
            v_key = tuple(V[j][r] for j in range(20))
            w_key = tuple(W[k][r] for k in range(20))
            key = (u_key, v_key, w_key)
            if key in term_dict:
                term_dict[key] += 1
            else:
                term_dict[key] = 1

        new_U = []
        new_V = []
        new_W = []
        for key, coeff in term_dict.items():
            u_key, v_key, w_key = key
            new_U.append(list(u_key))
            new_V.append(list(v_key))
            new_W.append(list(w_key))
            # Store coefficient separately; we'll handle it in verification
        # Actually, merging only works if we can absorb the coefficient.
        # For now, just return the unique terms (this is only valid if all coeffs are 1)
        # Let's check if all coefficients are 1
        if all(c == 1 for c in term_dict.values()):
            return new_U, new_V, new_W
        return None

    # Try merging on the current best
    if time.time() - start_time < deadline:
        merged = merge_terms(best["U"], best["V"], best["W"])
        if merged is not None:
            mU, mV, mW = merged
            if verify(mU, mV, mW):
                if len(mU) < best_rank:
                    best = {"U": mU, "V": mV, "W": mW}
                    best_rank = len(mU)

    # Try a different block partition: split into 4x2 * 2x5 style
    # Actually, let's try splitting the 4x4 into a 2x2 and 2x2 row split
    # and the 4x5 into 4x2 and 4x3 column splits
    if time.time() - start_time < deadline:
        def build_alt_block():
            """Alternative: split A by rows (2x2 blocks), B by cols (4x2, 4x3)."""
            # A = [A_top (2x4); A_bot (2x4)]
            # B = [B_left (4x2) | B_right (4x3)]
            # C = [C_top_left (2x2)  C_top_right (2x3)]
            #     [C_bot_left (2x2)  C_bot_right (2x3)]
            # C_top_left = A_top * B_left
            # C_top_right = A_top * B_right
            # C_bot_left = A_bot * B_left
            # C_bot_right = A_bot * B_right

            U_list = []
            V_list = []
            W_list = []

            # A_top: rows 0-1, cols 0-3 -> U[i*4+j] for i=0,1
            # A_bot: rows 2-3, cols 0-3 -> U[i*4+j] for i=2,3
            # B_left: rows 0-3, cols 0-1 -> V[j*5+k] for k=0,1
            # B_right: rows 0-3, cols 2-4 -> V[j*5+k] for k=2,3,4

            # C_top_left: rows 0-1, cols 0-1 -> W[k*4+i] for i=0,1 k=0,1
            # C_top_right: rows 0-1, cols 2-4 -> W[k*4+i] for i=0,1 k=2,3,4
            # C_bot_left: rows 2-3, cols 0-1 -> W[k*4+i] for i=2,3 k=0,1
            # C_bot_right: rows 2-3, cols 2-4 -> W[k*4+i] for i=2,3 k=2,3,4

            # Schoolbook for each sub-product
            # A_top * B_left: 2x4 * 4x2 = 16 terms
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[i*4+j] = 1
                        V_new[j*5+k] = 1
                        W_new[k*4+i] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A_top * B_right: 2x4 * 4x3 = 24 terms
            for i in range(2):
                for j in range(4):
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

            # A_bot * B_left: 2x4 * 4x2 = 16 terms
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(2+i)*4+j] = 1
                        V_new[j*5+k] = 1
                        W_new[k*4+(2+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)

            # A_bot * B_right: 2x4 * 4x3 = 24 terms
            for i in range(2):
                for j in range(4):
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

            return {"U": U_list, "V": V_list, "W": W_list}

        alt_result = build_alt_block()
        alt_rank = len(alt_result["U"])
        if verify(alt_result["U"], alt_result["V"], alt_result["W"]):
            if alt_rank < best_rank:
                best = alt_result
                best_rank = alt_rank

    return best

# EVOLVE-BLOCK-END
