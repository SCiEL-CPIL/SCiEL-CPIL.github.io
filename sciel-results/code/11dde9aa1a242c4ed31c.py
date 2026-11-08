import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    # U, V, W are 16x49 matrices
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)

    # Helper to map 2D indices to 1D
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i  # Trace convention: W[k*n+i, r] -> W[col*n + row]

    # Pre-calculate indices for blocks
    # Block (bi, bj) of A covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    def get_a_indices(bi, bj):
        return [
            u_idx(2*bi, 2*bj),
            u_idx(2*bi, 2*bj+1),
            u_idx(2*bi+1, 2*bj),
            u_idx(2*bi+1, 2*bj+1)
        ]

    def get_b_indices(bi, bj):
        return [
            v_idx(2*bi, 2*bj),
            v_idx(2*bi, 2*bj+1),
            v_idx(2*bi+1, 2*bj),
            v_idx(2*bi+1, 2*bj+1)
        ]

    A00_idx = get_a_indices(0, 0)
    A01_idx = get_a_indices(0, 1)
    A10_idx = get_a_indices(1, 0)
    A11_idx = get_a_indices(1, 1)

    B00_idx = get_b_indices(0, 0)
    B01_idx = get_b_indices(0, 1)
    B10_idx = get_b_indices(1, 0)
    B11_idx = get_b_indices(1, 1)

    # Convert index lists to combo dicts with coeff 1
    def to_combo(indices):
        return {i: 1 for i in indices}

    A00 = to_combo(A00_idx)
    A01 = to_combo(A01_idx)
    A10 = to_combo(A10_idx)
    A11 = to_combo(A11_idx)

    B00 = to_combo(B00_idx)
    B01 = to_combo(B01_idx)
    B10 = to_combo(B10_idx)
    B11 = to_combo(B11_idx)

    def add_combos(c1, c2):
        d = {}
        for idx, c in c1.items():
            d[idx] = d.get(idx, 0) + c
        for idx, c in c2.items():
            d[idx] = d.get(idx, 0) + c
        return {k: v for k, v in d.items() if v != 0}

    def sub_combos(c1, c2):
        d = {}
        for idx, c in c1.items():
            d[idx] = d.get(idx, 0) + c
        for idx, c in c2.items():
            d[idx] = d.get(idx, 0) - c
        return {k: v for k, v in d.items() if v != 0}

    # Define the 7 block-level Strassen terms
    # Each term: L (2x2 matrix of combos), R (2x2 matrix of combos), contrib (block output -> global C contributions)
    # Block output (pr, pc) means the (pr,pc) element of the 2x2 block product P contributes to global C

    L_terms = []
    R_terms = []
    contrib_terms = []

    # T0: P1 = (A00+A11) * (B00+B11)
    # P1 contributes to C00 (block 0,0) with +1 and C11 (block 1,1) with +1
    L0 = [
        add_combos(A00, A11),  # L[0][0]
        add_combos(A00, A11),  # L[0][1] -- wait, this is wrong
    ]
    # Actually, each element of L is a separate combo. Let me redo this properly.
    # L is a 2x2 matrix where L[r][c] is a linear combo of A elements.
    # For P1 = (A00+A11) * (B00+B11):
    # L[0][0] = A00[0] + A11[0] (first element of A00 block + first element of A11 block)
    # etc.

    # Let me define it element by element
    # Block A00 has elements at indices A00_idx[0..3]
    # Block A11 has elements at indices A11_idx[0..3]
    # L[0][0] = A00_idx[0] + A11_idx[0]
    # L[0][1] = A00_idx[1] + A11_idx[1]
    # L[1][0] = A00_idx[2] + A11_idx[2]
    # L[1][1] = A00_idx[3] + A11_idx[3]

    def make_L_from_blocks(block1_idx, block2_idx, sign1=1, sign2=1):
        """Create a 2x2 L matrix from two blocks with signs"""
        L = []
        for r in range(2):
            row = []
            for c in range(2):
                idx1 = block1_idx[r * 2 + c]
                idx2 = block2_idx[r * 2 + c]
                d = {idx1: sign1, idx2: sign2}
                # Remove zeros
                d = {k: v for k, v in d.items() if v != 0}
                row.append(d)
            L.append(row)
        return L

    def make_R_from_blocks(block1_idx, block2_idx, sign1=1, sign2=1):
        """Create a 2x2 R matrix from two blocks with signs"""
        R = []
        for r in range(2):
            row = []
            for c in range(2):
                idx1 = block1_idx[r * 2 + c]
                idx2 = block2_idx[r * 2 + c]
                d = {idx1: sign1, idx2: sign2}
                d = {k: v for k, v in d.items() if v != 0}
                row.append(d)
            R.append(row)
        return R

    def make_single_block_L(block_idx):
        """Create a 2x2 L matrix from a single block"""
        L = []
        for r in range(2):
            row = []
            for c in range(2):
                idx = block_idx[r * 2 + c]
                row.append({idx: 1})
            L.append(row)
        return L

    def make_single_block_R(block_idx):
        """Create a 2x2 R matrix from a single block"""
        R = []
        for r in range(2):
            row = []
            for c in range(2):
                idx = block_idx[r * 2 + c]
                row.append({idx: 1})
            R.append(row)
        return R

    # T0: P1 = (A00+A11) * (B00+B11)
    L_terms.append(make_L_from_blocks(A00_idx, A11_idx, 1, 1))
    R_terms.append(make_R_from_blocks(B00_idx, B11_idx, 1, 1))
    # P1 contributes to C00 (+1) and C11 (+1)
    # C00 is block (0,0), C11 is block (1,1)
    contrib0 = {
        (0, 0): [(0, 0, 1)],   # P1[0,0] -> C[0,0] with +1
        (0, 1): [(0, 0, 1)],   # P1[0,1] -> C[0,0] with +1? No wait...
    }

    # Let me think about this more carefully.
    # The block product P = L * R is a 2x2 matrix.
    # P[r][c] is the (r,c) element of the 2x2 block product.
    # This contributes to the global C matrix at the corresponding block position.
    # For T0: P1 contributes to C00 (block 0,0) and C11 (block 1,1).
    # P1[r][c] contributes to C_block[0,0][r][c] and C_block[1,1][r][c]
    # C_block[0,0][r][c] = C[2*0+r][2*0+c] = C[r][c]
    # C_block[1,1][r][c] = C[2*1+r][2*1+c] = C[2+r][2+c]

    # So for each element (r,c) of P1, it contributes to:
    # C[r][c] with coeff +1 and C[2+r][2+c] with coeff +1

    def make_contrib_block(block_row, block_col, coeff=1):
        """Create contribution mapping for a 2x2 block at (block_row, block_col)"""
        contrib = {}
        for r in range(2):
            for c in range(2):
                out_r = 2 * block_row + r
                out_c = 2 * block_col + c
                contrib[(r, c)] = [(out_r, out_c, coeff)]
        return contrib

    def combine_contribs(c1, c2):
        """Combine two contribution mappings"""
        combined = {}
        for key in set(list(c1.keys()) + list(c2.keys())):
            contributions = []
            if key in c1:
                contributions.extend(c1[key])
            if key in c2:
                contributions.extend(c2[key])
            # Combine contributions to same output
            out_dict = {}
            for (out_r, out_c, coeff) in contributions:
                out_dict[(out_r, out_c)] = out_dict.get((out_r, out_c), 0) + coeff
            combined[key] = [(r, c, v) for (r, c), v in out_dict.items() if v != 0]
        return combined

    # T0: P1 = (A00+A11) * (B00+B11)
    # Contributes to C00 (+1) and C11 (+1)
    contrib_terms.append(combine_contribs(make_contrib_block(0, 0, 1), make_contrib_block(1, 1, 1)))

    # T1: P2 = (A10+A11) * B00
    # Contributes to C10 (+1) and C11 (-1)
    L_terms.append(make_L_from_blocks(A10_idx, A11_idx, 1, 1))
    R_terms.append(make_single_block_R(B00_idx))
    contrib_terms.append(combine_contribs(make_contrib_block(1, 0, 1), make_contrib_block(1, 1, -1)))

    # T2: P3 = A00 * (B01-B11)
    # Contributes to C01 (+1) and C11 (+1)
    L_terms.append(make_single_block_L(A00_idx))
    R_terms.append(make_R_from_blocks(B01_idx, B11_idx, 1, -1))
    contrib_terms.append(combine_contribs(make_contrib_block(0, 1, 1), make_contrib_block(1, 1, 1)))

    # T3: P4 = A11 * (B10-B00)
    # Contributes to C00 (+1) and C10 (+1)
    L_terms.append(make_single_block_L(A11_idx))
    R_terms.append(make_R_from_blocks(B10_idx, B00_idx, 1, -1))
    contrib_terms.append(combine_contribs(make_contrib_block(0, 0, 1), make_contrib_block(1, 0, 1)))

    # T4: P5 = (A00+A01) * B11
    # Contributes to C00 (-1) and C01 (+1)
    L_terms.append(make_L_from_blocks(A00_idx, A01_idx, 1, 1))
    R_terms.append(make_single_block_R(B11_idx))
    contrib_terms.append(combine_contribs(make_contrib_block(0, 0, -1), make_contrib_block(0, 1, 1)))

    # T5: P6 = (A10-A00) * (B00+B01)
    # Contributes to C11 (+1)
    L_terms.append(make_L_from_blocks(A10_idx, A00_idx, 1, -1))
    R_terms.append(make_R_from_blocks(B00_idx, B01_idx, 1, 1))
    contrib_terms.append(make_contrib_block(1, 1, 1))

    # T6: P7 = (A01-A11) * (B10+B11)
    # Contributes to C00 (+1)
    L_terms.append(make_L_from_blocks(A01_idx, A11_idx, 1, -1))
    R_terms.append(make_R_from_blocks(B10_idx, B11_idx, 1, 1))
    contrib_terms.append(make_contrib_block(0, 0, 1))

    # Strassen sub-product contributions
    # M1 = (a+d)(e+h) -> P[0,0]: +1, P[1,1]: +1
    # M2 = (c+d)e    -> P[1,0]: +1, P[1,1]: -1
    # M3 = a(f-h)    -> P[0,1]: +1, P[1,1]: +1
    # M4 = d(g-e)    -> P[0,0]: +1, P[1,0]: +1
    # M5 = (a+b)h    -> P[0,0]: -1, P[0,1]: +1
    # M6 = (c-a)(e+f)-> P[1,1]: +1
    # M7 = (b-d)(g+h)-> P[0,0]: +1
    sub_contrib = [
        [(0, 0, 1), (1, 1, 1)],    # M1
        [(1, 0, 1), (1, 1, -1)],   # M2
        [(0, 1, 1), (1, 1, 1)],    # M3
        [(0, 0, 1), (1, 0, 1)],    # M4
        [(0, 0, -1), (0, 1, 1)],   # M5
        [(1, 1, 1)],               # M6
        [(0, 0, 1)]                # M7
    ]

    # For each term t (0-6) and each sub-product k (0-6):
    for t in range(7):
        L_t = L_terms[t]  # 2x2, each element is dict {u_idx: coeff}
        R_t = R_terms[t]  # 2x2, each element is dict {v_idx: coeff}
        contrib_t = contrib_terms[t]  # dict (pr,pc) -> list of (out_row, out_col, coeff)

        a = L_t[0][0]
        b = L_t[0][1]
        c = L_t[1][0]
        d = L_t[1][1]
        e = R_t[0][0]
        f = R_t[0][1]
        g = R_t[1][0]
        h = R_t[1][1]

        # Define the 7 sub-products
        sub_products = []

        # M1: X = a+d, Y = e+h
        x1 = add_combos(a, d)
        y1 = add_combos(e, h)
        sub_products.append((x1, y1))

        # M2: X = c+d, Y = e
        x2 = add_combos(c, d)
        y2 = dict(e)
        sub_products.append((x2, y2))

        # M3: X = a, Y = f-h
        x3 = dict(a)
        y3 = sub_combos(f, h)
        sub_products.append((x3, y3))

        # M4: X = d, Y = g-e
        x4 = dict(d)
        y4 = sub_combos(g, e)
        sub_products.append((x4, y4))

        # M5: X = a+b, Y = h
        x5 = add_combos(a, b)
        y5 = dict(h)
        sub_products.append((x5, y5))

        # M6: X = c-a, Y = e+f
        x6 = sub_combos(c, a)
        y6 = add_combos(e, f)
        sub_products.append((x6, y6))

        # M7: X = b-d, Y = g+h
        x7 = sub_combos(b, d)
        y7 = add_combos(g, h)
        sub_products.append((x7, y7))

        # Now for each sub-product k (0-6), add to U, V, W
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k

            # Add to U (from x_k, which are A elements)
            for u_index, coeff in x_k.items():
                U[u_index, rank_idx] += coeff

            # Add to V (from y_k, which are B elements)
            for v_index, coeff in y_k.items():
                V[v_index, rank_idx] += coeff

            # Add to W (contributions from sub_contrib[k] through contrib_t)
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_index = w_idx(out_row, out_col)
                        W[w_index, rank_idx] += sub_coeff * block_coeff

    # Verify the decomposition
    # Tensor equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    # Let's verify with a few random tests
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Quick verification with random matrices
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(4, 4))
        B_test = np.random.randint(-5, 6, size=(4, 4))
        C_expected = A_test @ B_test

        # Compute using the decomposition
        C_computed = np.zeros((4, 4), dtype=np.int64)
        for r in range(rank):
            # U_col = U[:, r] has 16 entries, representing A elements
            # V_col = V[:, r] has 16 entries, representing B elements
            # W_col = W[:, r] has 16 entries, representing C elements
            # The product U_col * V_col gives a scalar, which is added to C via W_col
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]

            # u_col[i*4+j] is the coefficient for A[i,j]
            # v_col[j*4+k] is the coefficient for B[j,k]
            # w_col[k*4+i] is the coefficient for C[i,k]

            # The scalar product is sum_{i,j} u_col[i*4+j] * A[i,j] * sum_{j,k} v_col[j*4+k] * B[j,k]
            # Wait, that's not right. The tensor decomposition is:
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # So for fixed r, the contribution is U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This means for each r, we compute a rank-1 tensor.

            # Let's compute it differently:
            # For each r, the contribution to C[i,k] is:
            # sum_j U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]

            for i in range(4):
                for k in range(4):
                    w_val = W[k * 4 + i, r]
                    if w_val == 0:
                        continue
                    s = 0
                    for j in range(4):
                        u_val = U[i * 4 + j, r]
                        v_val = V[j * 4 + k, r]
                        s += u_val * v_val
                    C_computed[i, k] += w_val * s

        if not np.array_equal(C_computed, C_expected):
            # Verification failed - this shouldn't happen with correct Strassen
            # Fall back to a simple rank-64 decomposition
            U_fallback = np.zeros((16, 64), dtype=np.int64)
            V_fallback = np.zeros((16, 64), dtype=np.int64)
            W_fallback = np.zeros((16, 64), dtype=np.int64)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        idx = i * 16 + j * 4 + k
                        U_fallback[i * 4 + j, idx] = 1
                        V_fallback[j * 4 + k, idx] = 1
                        W_fallback[k * 4 + i, idx] = 1
            return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
