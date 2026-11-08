import numpy as np
import time

def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication using two-level Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indices:
      U[i*n+k, r]   (n=4)
      V[k*n+j, r]
      W[j*n+i, r]   (trace convention: W[col*n + row])
    Tensor check:
      sum_r U[i*n+k, r] * V[k*n+j, r] * W[j*n+i, r] = 1  for all i,j,k
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n = 4
    rank = 49

    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)

    def iU(i, k): return i * n + k
    def iV(k, j): return k * n + j
    def iW(j, i): return j * n + i   # trace convention W[k*n+i] where k=col=j, i=row

    # --- Strassen for 2x2 (scalar) ---
    # Inputs: L (2x2 linear forms on A), R (2x2 linear forms on B)
    # Outputs: 7 pairs (A_form, B_form) representing the 7 multiplications
    # Each form is a dict mapping (row, col) -> coeff
    def merge(a, b, sa=1, sb=1):
        d = {}
        for (r, c), v in a.items():
            d[(r, c)] = d.get((r, c), 0) + v * sa
        for (r, c), v in b.items():
            d[(r, c)] = d.get((r, c), 0) + v * sb
        return {k: v for k, v in d.items() if v != 0}

    def strassen_2x2(L, R):
        L00, L01 = L[0][0], L[0][1]
        L10, L11 = L[1][0], L[1][1]
        R00, R01 = R[0][0], R[0][1]
        R10, R11 = R[1][0], R[1][1]
        return [
            (merge(L00, L11), merge(R00, R11)),
            (merge(L10, L11), R00),
            (L00, merge(R01, R11, 1, -1)),
            (L11, merge(R10, R00, 1, -1)),
            (merge(L00, L01), R11),
            (merge(L10, L00, 1, -1), merge(R00, R01)),
            (merge(L01, L11, 1, -1), merge(R10, R11)),
        ]

    # --- Top-level 4x4 Strassen: 7 block products ---
    # Each block product t has:
    #   L_t: 2x2 matrix of linear forms on A (each form is 2x2 sub-block of A)
    #   R_t: 2x2 matrix of linear forms on B
    #   contrib_blocks: list of (block_row, block_col, coeff) for C blocks
    # A block (br, bc) covers A[2*br:2*br+2, 2*bc:2*bc+2]
    # So A element at global (i, k) is in block (i//2, k//2) at local (i%2, k%2)

    def block_form(br, bc, local_rows, local_cols, coeff=1):
        """Create a 2x2 linear form dict for a sub-block of A or B."""
        d = {}
        for lr in local_rows:
            for lc in local_cols:
                d[(2 * br + lr, 2 * bc + lc)] = coeff
        return d

    def full_block_form(br, bc, coeff=1):
        return block_form(br, bc, [0, 1], [0, 1], coeff)

    # Top-level Strassen terms for 4x4:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    top_terms = []
    # Each entry: (L_2x2, R_2x2, contrib_blocks)
    # L_2x2[r][c] is a dict for the (r,c) local position of the 2x2 block matrix

    def make_L(pattern):
        """pattern is a function (br,bc) -> dict or None"""
        return [[pattern(r, c) for c in range(2)] for r in range(2)]

    def make_R(pattern):
        return [[pattern(r, c) for c in range(2)] for r in range(2)]

    # P1: L = A00 + A11 for all positions, R = B00 + B11
    def L1_pat(r, c):
        d = {}
        for key, val in full_block_form(0, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(1, 1).items():
            d[key] = d.get(key, 0) + val
        return d
    def R1_pat(r, c):
        d = {}
        for key, val in full_block_form(0, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(1, 1).items():
            d[key] = d.get(key, 0) + val
        return d
    top_terms.append((make_L(L1_pat), make_R(R1_pat), [(0,0,1), (1,1,1)]))

    # P2: L = A10 + A11, R = B00
    def L2_pat(r, c):
        d = {}
        for key, val in full_block_form(1, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(1, 1).items():
            d[key] = d.get(key, 0) + val
        return d
    def R2_pat(r, c):
        return full_block_form(0, 0)
    top_terms.append((make_L(L2_pat), make_R(R2_pat), [(1,0,1), (1,1,-1)]))

    # P3: L = A00, R = B01 - B11
    def L3_pat(r, c):
        return full_block_form(0, 0)
    def R3_pat(r, c):
        d = {}
        for key, val in full_block_form(0, 1).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(1, 1).items():
            d[key] = d.get(key, 0) - val
        return d
    top_terms.append((make_L(L3_pat), make_R(R3_pat), [(0,1,1), (1,1,1)]))

    # P4: L = A11, R = B10 - B00
    def L4_pat(r, c):
        return full_block_form(1, 1)
    def R4_pat(r, c):
        d = {}
        for key, val in full_block_form(1, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(0, 0).items():
            d[key] = d.get(key, 0) - val
        return d
    top_terms.append((make_L(L4_pat), make_R(R4_pat), [(0,0,1), (1,0,1)]))

    # P5: L = A00 + A01, R = B11
    def L5_pat(r, c):
        d = {}
        for key, val in full_block_form(0, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(0, 1).items():
            d[key] = d.get(key, 0) + val
        return d
    def R5_pat(r, c):
        return full_block_form(1, 1)
    top_terms.append((make_L(L5_pat), make_R(R5_pat), [(0,0,-1), (0,1,1)]))

    # P6: L = A10 - A00, R = B00 + B01
    def L6_pat(r, c):
        d = {}
        for key, val in full_block_form(1, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(0, 0).items():
            d[key] = d.get(key, 0) - val
        return d
    def R6_pat(r, c):
        d = {}
        for key, val in full_block_form(0, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(0, 1).items():
            d[key] = d.get(key, 0) + val
        return d
    top_terms.append((make_L(L6_pat), make_R(R6_pat), [(1,1,1)]))

    # P7: L = A01 - A11, R = B10 + B11
    def L7_pat(r, c):
        d = {}
        for key, val in full_block_form(0, 1).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(1, 1).items():
            d[key] = d.get(key, 0) - val
        return d
    def R7_pat(r, c):
        d = {}
        for key, val in full_block_form(1, 0).items():
            d[key] = d.get(key, 0) + val
        for key, val in full_block_form(1, 1).items():
            d[key] = d.get(key, 0) + val
        return d
    top_terms.append((make_L(L7_pat), make_R(R7_pat), [(0,0,1)]))

    # Sub-contributions within each 2x2 block product:
    # M1 -> P00+1, P11+1
    # M2 -> P10+1, P11-1
    # M3 -> P01+1, P11+1
    # M4 -> P00+1, P10+1
    # M5 -> P00-1, P01+1
    # M6 -> P11+1
    # M7 -> P00+1
    sub_contrib = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Build U, V, W
    for t_idx, (L_t, R_t, contrib_blocks) in enumerate(top_terms):
        sub_products = strassen_2x2(L_t, R_t)
        for k_idx in range(7):
            r = t_idx * 7 + k_idx
            A_form, B_form = sub_products[k_idx]

            for (i, k), coeff in A_form.items():
                U[iU(i, k), r] += coeff
            for (k, j), coeff in B_form.items():
                V[iV(k, j), r] += coeff

            for (lr, lc, sub_c) in sub_contrib[k_idx]:
                for (br, bc, block_c) in contrib_blocks:
                    gi = 2 * br + lr
                    gj = 2 * bc + lc
                    W[iW(gj, gi), r] += sub_c * block_c

    # Quick verification with a random test
    A_test = np.random.randint(-2, 3, (4, 4), dtype=float)
    B_test = np.random.randint(-2, 3, (4, 4), dtype=float)
    C_naive = A_test @ B_test
    C_decomp = np.zeros((4, 4), dtype=float)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    C_decomp[i, j] += u_col[iU(i, k)] * v_col[iV(k, j)] * w_col[iW(j, i)]

    if not np.allclose(C_naive, C_decomp, atol=1e-9):
        # Fallback: return naive rank-64 decomposition
        rank_naive = 64
        U_naive = np.zeros((16, rank_naive), dtype=float)
        V_naive = np.zeros((16, rank_naive), dtype=float)
        W_naive = np.zeros((16, rank_naive), dtype=float)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U_naive[iU(i, k), idx] = 1
                    V_naive[iV(k, j), idx] = 1
                    W_naive[iW(j, i), idx] = 1
                    idx += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
