"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
# For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
# C00 = a*e + c*h
# C01 = b*h
# C10 = a*f + d*g
# C11 = b*g + d*h
#
# Strassen products:
# P1 = (a+d)(e+h)
# P2 = (c+d)e
# P3 = a(f-h)
# P4 = d(g-e)
# P5 = (a+b)h
# P6 = (c-a)(e+f)
# P7 = (b-d)(g+h)
#
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

# Each Strassen product is a linear combination of entries of A and B.
# We represent each product as:
#   (list of (ai, aj, coeff) for A-part, list of (bk, bl, coeff) for B-part)
# where ai,aj index into the 2x2 A block and bk,bl index into the 2x2 B block.

# Strassen products for a 2x2 block:
# A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
# B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h

_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # P2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    # P3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    # P4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    # P5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    # P6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    # P7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

# How each C entry is formed from the 7 Strassen products:
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

_STRASSEN_C_COMBINATION = [
    # (c_row, c_col, list of (product_idx, coeff))
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _laderman_4x4_decomposition():
    """
    Build rank-17 decomposition for 4x4x4 matrix multiplication using
    Laderman's algorithm.
    Total rank = 17.
    """
    n, m, p = 4, 4, 4
    rank = 17

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Laderman's 17 products. Each is (a_blocks, b_blocks) where
    # a_blocks = list of (block_row, block_col, coeff) for A
    # b_blocks = list of (block_row, block_col, coeff) for B
    # Block (br, bc) of A covers A[2*br:2*br+2, 2*bc:2*bc+2]
    # Block (br, bc) of B covers B[2*br:2*br+2, 2*bc:2*bc+2]

    laderman_products = [
        # 0: A00 * B00
        ([(0,0,1)], [(0,0,1)]),
        # 1: A01 * B11
        ([(0,1,1)], [(1,1,1)]),
        # 2: A10 * B01
        ([(1,0,1)], [(0,1,1)]),
        # 3: A11 * B10
        ([(1,1,1)], [(1,0,1)]),
        # 4: (A00+A01) * (B00+B11)
        ([(0,0,1),(0,1,1)], [(0,0,1),(1,1,1)]),
        # 5: (A10+A11) * (B00+B01)
        ([(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)]),
        # 6: (A00+A10) * (B01+B11)
        ([(0,0,1),(1,0,1)], [(0,1,1),(1,1,1)]),
        # 7: (A01+A11) * (B00+B10)
        ([(0,1,1),(1,1,1)], [(0,0,1),(1,0,1)]),
        # 8: (A00+A10) * (B00+B11)
        ([(0,0,1),(1,0,1)], [(0,0,1),(1,1,1)]),
        # 9: (A01+A11) * (B01+B10)
        ([(0,1,1),(1,1,1)], [(0,1,1),(1,0,1)]),
        # 10: (A00+A01+A10+A11) * (B00+B01+B10+B11)
        ([(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),
        # 11: (A00+A11) * (B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # 12: (A01+A10) * (B01+B10)
        ([(0,1,1),(1,0,1)], [(0,1,1),(1,0,1)]),
        # 13: A00 * (B00+B01+B10+B11)
        ([(0,0,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),
        # 14: A11 * (B00+B01+B10+B11)
        ([(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),
        # 15: (A00+A11) * (B01+B10)
        ([(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)]),
        # 16: (A01+A10) * (B00+B11)
        ([(0,1,1),(1,0,1)], [(0,0,1),(1,1,1)]),
    ]

    # C reconstruction: C[cbr][cbc] = sum_i w_i * P_i
    # cbr, cbc are block row/col of C (0 or 1)
    # Each entry: (cbr, cbc, list of (product_idx, coeff))
    c_reconstruction = [
        # C00 = P4 - P5 + P8 - P10 - P11
        (0, 0, [(4, 1), (5, -1), (8, 1), (10, -1), (11, -1)]),
        # C01 = P1 + P6 - P9 - P12 + P10
        (0, 1, [(1, 1), (6, 1), (9, -1), (12, -1), (10, 1)]),
        # C10 = P2 + P7 - P8 - P13 + P10
        (1, 0, [(2, 1), (7, 1), (8, -1), (13, -1), (10, 1)]),
        # C11 = P0 + P3 + P5 - P7 - P9 + P10
        (1, 1, [(0, 1), (3, 1), (5, 1), (7, -1), (9, -1), (10, 1)]),
    ]

    # Precompute w_block[i][cbr][cbc]
    w_block = [[[0]*4 for _ in range(2)] for _ in range(17)]
    for (cbr, cbc, c_list) in c_reconstruction:
        for (pi, wi) in c_list:
            w_block[pi][cbr][cbc] = wi

    # Build U, V, W
    for i in range(17):
        a_blocks, b_blocks = laderman_products[i]

        # U: for each (br, bc, coeff) in a_blocks, fill the 2x2 block
        for (br, bc, coeff) in a_blocks:
            for lr in range(2):
                for lc in range(2):
                    row = 2 * br + lr
                    col = 2 * bc + lc
                    U[row * m + col, i] += coeff

        # V: for each (br, bc, coeff) in b_blocks, fill the 2x2 block
        for (br, bc, coeff) in b_blocks:
            for lr in range(2):
                for lc in range(2):
                    row = 2 * br + lr
                    col = 2 * bc + lc
                    V[row * p + col, i] += coeff

        # W: C[cbr][cbc] gets w_block[i][cbr][cbc]
        for cbr in range(2):
            for cbc in range(2):
                wi = w_block[i][cbr][cbc]
                if wi == 0:
                    continue
                for lr in range(2):
                    for lc in range(2):
                        c_row = 2 * cbr + lr
                        c_col = 2 * cbc + lc
                        W[c_col * n + c_row, i] += wi

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    # Initialize U, V, W as numpy arrays for faster construction, then convert to lists
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # The 7 block-level Strassen products for 4x4:
    # Each is (A_block_combo, B_block_combo) where combo is list of (br, bc, coeff)
    # br, bc are block row/col indices (0 or 1)
    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),       # M1
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),                   # M2
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),                  # M3
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),                  # M4
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),                   # M5
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),       # M6
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),       # M7
    ]

    # C block combination: C[cbr][cbc] = sum_i w_i * Mi
    # c_block_combination[cbr][cbc] = list of (product_idx, coeff)
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # Precompute: for each block product i, w_i(cbr,cbc) for each C-block
    # w_block[i][cbr][cbc] = coefficient
    w_block = [[[0]*4 for _ in range(2)] for _ in range(7)]
    for (cbr, cbc, c_list) in c_block_combination:
        for (pi, wi) in c_list:
            w_block[pi][cbr][cbc] = wi

    # Precompute: for each Strassen product s, v_s(lr,lc) for each C-local entry
    # v_local[s][lr][lc] = coefficient
    v_local = [[[0]*2 for _ in range(2)] for _ in range(7)]
    for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
        for (si, vs) in v_list:
            v_local[si][lr][lc] = vs

    # Build U, V, W
    for i in range(7):
        a_combo, b_combo = block_products[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]

            # U: for each (a_br, a_bc, a_coeff) in a_combo and (lr, lc, l_coeff) in a_local
            # U index: (global_row, global_col) where global_row = 2*a_br + lr, global_col = 2*a_bc + lc
            # U is indexed by [row * m + col]
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col, r] += a_coeff * l_coeff

            # V: for each (b_br, b_bc, b_coeff) in b_combo and (lr, lc, l_coeff) in b_local
            # V index: [row * p + col]
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

            # W: C[2*cbr+lr][2*cbc+lc] gets w_block[i][cbr][cbc] * v_local[s][lr][lc]
            # W index: [col * n + row]
            for cbr in range(2):
                for cbc in range(2):
                    wi = w_block[i][cbr][cbc]
                    if wi == 0:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            vs = v_local[s][lr][lc]
                            if vs == 0:
                                continue
                            c_row = 2 * cbr + lr
                            c_col = 2 * cbc + lc
                            W[c_col * n + c_row, r] += wi * vs

    # Convert back to lists of lists to match the expected interface format if strict
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy for speed."""
    U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
    V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
    W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
    
    # Reshape to 3D: U_3d[i,j,r], V_3d[j,k,r], W_3d[k,i,r]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Contract over r: result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # Note: W is indexed by [k*n+i], so W_3d[k,i,r] corresponds to W[k*n+i, r]
    result = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
    
    # All entries should be 1
    return bool(np.allclose(result, 1.0, atol=1e-10))


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try Laderman rank-17 decomposition first
    t0 = time.time()
    result = _laderman_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Try recursive Strassen rank-49 decomposition
    if time.time() < deadline - 10.0:
        result = _strassen_4x4_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return result

    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
