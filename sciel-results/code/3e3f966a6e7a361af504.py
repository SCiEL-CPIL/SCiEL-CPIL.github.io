import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 135.0

    N, M, P = 4, 4, 5

    # --- Verification helper ---
    def check_validity(Umat, Vmat, Wmat):
        r = Umat.shape[1]
        if Vmat.shape != (M * P, r) or Wmat.shape != (P * N, r):
            return False
        U3 = Umat.reshape(N, M, r)
        V3 = Vmat.reshape(M, P, r)
        W3 = Wmat.reshape(P, N, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return bool(np.allclose(T, 1.0, atol=1e-10))

    # --- Schoolbook fallback (rank 80) ---
    def schoolbook():
        R = N * M * P
        U = np.zeros((N * M, R))
        V = np.zeros((M * P, R))
        W = np.zeros((P * N, R))
        idx = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, idx] = 1.0
                    V[j * P + k, idx] = 1.0
                    W[k * N + i, idx] = 1.0
                    idx += 1
        return U, V, W

    # --- Rank-65 construction: 49 (Strassen 4x4x4) + 16 (tail) ---
    RANK = 65
    U = np.zeros((N * M, RANK))
    V = np.zeros((M * P, RANK))
    W = np.zeros((P * N, RANK))

    # Strassen 2x2 multiplication pattern (7 terms)
    # Each term: (u_pattern, v_pattern, w_pattern)
    # u_pattern: 2x2 matrix of coefficients for A blocks
    # v_pattern: 2x2 matrix of coefficients for B blocks  
    # w_pattern: list of (c_row, c_col, coeff) for C blocks

    # The 7 Strassen terms at the 2x2 level
    S2 = [
        # (U_block_coeffs, V_block_coeffs, [(c_r, c_c, coeff), ...])
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [(0,0,1), (1,1,1)]),   # M1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [(1,0,1), (1,1,-1)]),  # M2
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [(0,1,1), (1,1,1)]),  # M3
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [(0,0,1), (1,0,1)]),  # M4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [(0,0,-1), (0,1,1)]),  # M5
        ([[0, -1], [1, 0]], [[1, 1], [0, 0]], [(1,1,1)]),           # M6
        ([[0, 1], [0, -1]], [[0, 1], [0, 1]], [(0,0,1)]),           # M7
    ]

    # 4x4x4 via recursive Strassen:
    # Split 4x4 into 2x2 blocks of 2x2 sub-blocks
    # Outer level: 7 terms, each operating on 2x2 blocks
    # Inner level: each 2x2 block multiplication uses 7 Strassen terms
    # Total: 7 * 7 = 49 terms

    # Outer Strassen terms: define how they combine the 4 2x2 blocks
    # Block indexing: C00=top-left, C01=top-right, C10=bottom-left, C11=bottom-right
    # Each block is a 2x2 matrix of scalar entries

    # For the outer level, we need to express U_outer and V_outer as 
    # linear combinations of the 4 2x2 blocks of A and B respectively.

    # Outer term definitions: (U_block_combos, V_block_combos, C_block_targets)
    # U_block_combos[r][c] = dict mapping (ar, ac) -> coeff, where (ar,ac) is block position
    # Similarly for V_block_combos
    # C_block_targets: list of (target_block, coeff) where target_block is (br, bc)

    # Actually, let me use a cleaner formulation.
    # The outer Strassen has 7 terms. For each term t:
    #   U_t is a 2x2 matrix where each entry is a 2x2 linear combo of A's 2x2 blocks
    #   V_t is a 2x2 matrix where each entry is a 2x2 linear combo of B's 2x2 blocks
    #   The product U_t * V_t (using 2x2 matrix mult) gives a 2x2 result
    #   Each of the 4 entries of the result corresponds to a C block

    # Let me define the outer 7 terms explicitly as 4x4 coefficient matrices
    # for U and V, where the 4x4 represents the full A and B matrices.

    # Outer term 1: U = A00 + A11, V = B00 + B11
    # In 4x4 index: A00 = rows 0-1, cols 0-1; A11 = rows 2-3, cols 2-3
    # U1 is 4x4: has 1s at (0,0),(0,1),(1,0),(1,1) [A00] and (2,2),(2,3),(3,2),(3,3) [A11]
    # V1 is 4x4: same pattern for B

    def make_4x4_block_sum(block_positions):
        """Create 4x4 matrix with 1s in the specified 2x2 block positions"""
        mat = np.zeros((4, 4))
        for (br, bc) in block_positions:
            mat[2*br:2*br+2, 2*bc:2*bc+2] = 1.0
        return mat

    # Define the 7 outer terms as 4x4 matrices
    # Block positions: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right

    U_outer = []
    V_outer = []
    # C_block targets for each outer term: dict mapping (c_br, c_bc) -> coeff
    C_targets = []

    # Term 1: (A00+A11)(B00+B11) -> contributes to C00 and C11
    U_outer.append(make_4x4_block_sum([(0,0), (1,1)]))
    V_outer.append(make_4x4_block_sum([(0,0), (1,1)]))
    C_targets.append({(0,0): 1, (1,1): 1})

    # Term 2: (A10+A11)B00 -> contributes to C10 and C11 (with -1 for C11)
    U_outer.append(make_4x4_block_sum([(1,0), (1,1)]))
    V_outer.append(make_4x4_block_sum([(0,0)]))
    C_targets.append({(1,0): 1, (1,1): -1})

    # Term 3: A00(B01-B11) -> contributes to C01 and C11
    U_outer.append(make_4x4_block_sum([(0,0)]))
    V3 = make_4x4_block_sum([(0,1)]) - make_4x4_block_sum([(1,1)])
    V_outer.append(V3)
    C_targets.append({(0,1): 1, (1,1): 1})

    # Term 4: A11(B10-B00) -> contributes to C00 and C10
    U_outer.append(make_4x4_block_sum([(1,1)]))
    V4 = make_4x4_block_sum([(1,0)]) - make_4x4_block_sum([(0,0)])
    V_outer.append(V4)
    C_targets.append({(0,0): 1, (1,0): 1})

    # Term 5: (A00+A01)B11 -> contributes to C00 (-1) and C01
    U_outer.append(make_4x4_block_sum([(0,0), (0,1)]))
    V_outer.append(make_4x4_block_sum([(1,1)]))
    C_targets.append({(0,0): -1, (0,1): 1})

    # Term 6: (A10-A00)(B00+B01) -> contributes to C11
    U6 = make_4x4_block_sum([(1,0)]) - make_4x4_block_sum([(0,0)])
    U_outer.append(U6)
    V_outer.append(make_4x4_block_sum([(0,0), (0,1)]))
    C_targets.append({(1,1): 1})

    # Term 7: (A01-A11)(B10+B11) -> contributes to C00
    U7 = make_4x4_block_sum([(0,1)]) - make_4x4_block_sum([(1,1)])
    U_outer.append(U7)
    V_outer.append(make_4x4_block_sum([(1,0), (1,1)]))
    C_targets.append({(0,0): 1})

    # Now, for each outer term j, we need to further decompose the 2x2 block 
    # multiplication using inner Strassen (7 terms each).
    # 
    # For outer term j, U_outer[j] is a 4x4 matrix. We can view it as a 2x2 
    # matrix of 2x2 blocks. Each block is a linear combination of the scalar 
    # entries of A. Similarly for V_outer[j].
    #
    # The product of these 2x2 block matrices (using 2x2 matrix multiplication) 
    # gives a 2x2 result where each entry is a scalar linear combination.
    #
    # Instead of doing this symbolically, let me directly compute the scalar 
    # coefficients by expanding.

    # For each outer term j:
    #   Let Uj_blocks be a 2x2 array of 2x2 matrices (each 2x2 matrix has the 
    #   scalar coefficients for A)
    #   Let Vj_blocks be a 2x2 array of 2x2 matrices (for B)
    #   
    #   The 2x2 block product: (Uj * Vj)[r][c] = sum_d Uj[r][d] @ Vj[d][c]
    #   where @ is 2x2 matrix multiplication.
    #
    #   Each resulting 2x2 matrix entry corresponds to a scalar coefficient 
    #   for the C block.

    # Let me extract the 2x2 block structure from each 4x4 matrix
    def extract_blocks(mat4x4):
        """Extract 2x2 blocks from a 4x4 matrix as a 2x2 array of 2x2 matrices"""
        blocks = [[mat4x4[2*r:2*r+2, 2*c:2*c+2].copy() for c in range(2)] for r in range(2)]
        return blocks

    # Inner Strassen terms for 2x2 block multiplication
    # Same 7 terms as S2 above, but now operating on 2x2 matrices of scalars
    inner_terms = [
        # (u_coeffs, v_coeffs, c_coeffs)
        # u_coeffs: 2x2 matrix of 2x2 matrices (coefficients for the 4 scalar positions)
        # Actually, let me just use the S2 pattern directly.
    ]

    # For the inner 2x2 Strassen, the 7 terms are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Each "Ai" and "Bi" here is a 2x2 matrix of scalars.
    # The result is a 2x2 matrix of scalars that gets added to the appropriate C block.

    # Let me define the inner Strassen as a function that takes two 2x2x2x2 arrays
    # (each representing a 2x2 matrix of 2x2 scalar matrices) and returns 7 terms.

    def inner_strassen(Ublocks, Vblocks):
        """
        Ublocks: 2x2 array of 2x2 matrices (U[br][bc] is a 2x2 matrix)
        Vblocks: 2x2 array of 2x2 matrices
        Returns list of 7 (u_term, v_term) where u_term and v_term are 
        4x4 matrices representing the scalar coefficients.
        """
        # Flatten to get individual 2x2 blocks
        A00 = Ublocks[0][0]  # 2x2
        A01 = Ublocks[0][1]
        A10 = Ublocks[1][0]
        A11 = Ublocks[1][1]
        B00 = Vblocks[0][0]
        B01 = Vblocks[0][1]
        B10 = Vblocks[1][0]
        B11 = Vblocks[1][1]

        terms = []

        # M1 = (A00+A11)(B00+B11)
        u1 = A00 + A11
        v1 = B00 + B11
        terms.append((u1, v1))

        # M2 = (A10+A11)B00
        u2 = A10 + A11
        v2 = B00
        terms.append((u2, v2))

        # M3 = A00(B01-B11)
        u3 = A00
        v3 = B01 - B11
        terms.append((u3, v3))

        # M4 = A11(B10-B00)
        u4 = A11
        v4 = B10 - B00
        terms.append((u4, v4))

        # M5 = (A00+A01)B11
        u5 = A00 + A01
        v5 = B11
        terms.append((u5, v5))

        # M6 = (A10-A00)(B00+B01)
        u6 = A10 - A00
        v6 = B00 + B01
        terms.append((u6, v6))

        # M7 = (A01-A11)(B10+B11)
        u7 = A01 - A11
        v7 = B10 + B11
        terms.append((u7, v7))

        return terms

    # Inner Strassen contribution to C blocks
    # For each of the 7 inner terms, which C blocks does it contribute to, and with what coefficients?
    # This is the same as the standard Strassen reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    inner_C_contrib = {
        0: [(0,0, 1), (1,1, 1)],      # M1 -> C00 and C11
        1: [(1,0, 1), (1,1, -1)],     # M2 -> C10 and C11
        2: [(0,1, 1), (1,1, 1)],      # M3 -> C01 and C11
        3: [(0,0, 1), (1,0, 1)],      # M4 -> C00 and C10
        4: [(0,0, -1), (0,1, 1)],     # M5 -> C00 and C01
        5: [(1,1, 1)],                # M6 -> C11
        6: [(0,0, 1)],                # M7 -> C00
    }

    # Now build the full decomposition
    col = 0
    for j in range(7):
        # Extract 2x2 blocks from the 4x4 outer matrices
        Ub = extract_blocks(U_outer[j])
        Vb = extract_blocks(V_outer[j])

        # Get the 7 inner Strassen terms
        inner_terms = inner_strassen(Ub, Vb)

        for t in range(7):
            u_term, v_term = inner_terms[t]
            # u_term and v_term are 2x2 matrices of scalars (the linear combinations)
            # u_term represents the coefficients for A's scalar entries
            # v_term represents the coefficients for B's scalar entries

            # Add to U and V matrices
            for ar in range(2):
                for ac in range(2):
                    coeff = u_term[ar, ac]
                    if abs(coeff) > 1e-12:
                        a_row = ar
                        a_col = ac
                        U[a_row * M + a_col, col] += coeff

            for br in range(2):
                for bc in range(2):
                    coeff = v_term[br, bc]
                    if abs(coeff) > 1e-12:
                        b_row = br
                        b_col = bc
                        V[b_row * P + b_col, col] += coeff

            # Determine W contributions
            # The inner term t contributes to certain C blocks (from inner_C_contrib)
            # Each C block is further scaled by the outer C_targets
            w_coeffs = {}
            for (c_br, c_bc, inner_coeff) in inner_C_contrib[t]:
                # c_br, c_bc is the position within the 2x2 block structure
                # The outer C_targets tells us which top-level C blocks this maps to
                outer_coeff = C_targets[j].get((c_br, c_bc), 0)
                total_coeff = inner_coeff * outer_coeff
                if abs(total_coeff) > 1e-12:
                    # Map (c_br, c_bc) to the full 4x4 C matrix position
                    c_row_block = c_br
                    c_col_block = c_bc
                    # The C block at (c_row_block, c_col_block) occupies:
                    # rows 2*c_row_block to 2*c_row_block+1, cols 2*c_col_block to 2*c_col_block+1
                    for dr in range(2):
                        for dc in range(2):
                            c_row = 2 * c_row_block + dr
                            c_col = 2 * c_col_block + dc
                            w_idx = c_col * N + c_row  # W[k*n+i, r] convention: index = c_col * N + c_row
                            w_coeffs[w_idx] = w_coeffs.get(w_idx, 0) + total_coeff

            for w_idx, coeff in w_coeffs.items():
                if abs(coeff) > 1e-12:
                    W[w_idx, col] += coeff

            col += 1

    # Add 16 terms for the 5th column (k=4)
    for i in range(N):
        for j in range(M):
            U[i * M + j, col] = 1.0
            V[j * P + 4, col] = 1.0
            W[4 * N + i, col] = 1.0
            col += 1

    # Verify
    if check_validity(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    if time.time() < DEADLINE:
        Us, Vs, Ws = schoolbook()
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Last resort: return whatever we have
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
