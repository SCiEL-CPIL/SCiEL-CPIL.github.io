import numpy as np
import time

# ---------------------------------------------------------------------------
# Task: exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
# Rank = number of rank-1 terms. The decomposition must satisfy:
#   C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# (official trace convention: W indexed by [k*n + i]).
# ---------------------------------------------------------------------------

def construct():
    """
    Returns a dict with keys 'U', 'V', 'W' giving the factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For a 4x4 @ 4x5 multiplication we use the rank-49 two-level Strassen
    decomposition on the 4x4x4 tensor (rank 49) plus 16 extra rank-1 terms
    for the extra 5th column of B, giving total rank 65.
    """
    n, m, p = 4, 4, 5

    # ---- Build the 4x4x4 Strassen decomposition (rank 49) ----
    # We use a two-level Strassen: outer level splits 4x4x4 into 2x2x2 blocks
    # of size 2x2x2, and inner level uses the 2x2x2 Strassen algorithm.

    # The 2x2x2 Strassen algorithm uses 7 multiplications.
    # The 7 "intermediate" products are:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11) B00
    #   M3 = A00 (B01-B11)
    #   M4 = A11 (B10-B00)
    #   M5 = (A00+A10) B11
    #   M6 = (A01-A11) (B10+B11)
    #   M7 = (A10-A00) (B00+B01)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Inner 2x2x2 Strassen factor matrices (each of shape (4,7) for U/V/W):
    # U maps 2x2 A-block -> 7 products, V maps 2x2 B-block -> 7 products,
    # W maps 2x2 C-block -> 7 products (with signs).
    def _inner_strassen():
        U_in = np.zeros((4, 7), dtype=int)
        V_in = np.zeros((4, 7), dtype=int)
        W_in = np.zeros((4, 7), dtype=int)
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        # M1 = (A00+A11)(B00+B11)
        U_in[0, 0] = 1; U_in[3, 0] = 1
        V_in[0, 0] = 1; V_in[3, 0] = 1
        W_in[0, 0] = 1  # C00 gets +M1
        W_in[3, 0] = 1  # C11 gets +M1
        # M2 = (A10+A11) B00
        U_in[2, 1] = 1; U_in[3, 1] = 1
        V_in[0, 1] = 1
        W_in[2, 1] = 1  # C10 gets +M2
        W_in[3, 1] = -1 # C11 gets -M2
        # M3 = A00 (B01-B11)
        U_in[0, 2] = 1
        V_in[1, 2] = 1; V_in[3, 2] = -1
        W_in[1, 2] = 1  # C01 gets +M3
        W_in[3, 2] = 1  # C11 gets +M3
        # M4 = A11 (B10-B00)
        U_in[3, 3] = 1
        V_in[2, 3] = 1; V_in[0, 3] = -1
        W_in[0, 3] = 1  # C00 gets +M4
        W_in[2, 3] = 1  # C10 gets +M4
        # M5 = (A00+A10) B11
        U_in[0, 4] = 1; U_in[2, 4] = 1
        V_in[3, 4] = 1
        W_in[0, 4] = -1 # C00 gets -M5
        W_in[1, 4] = 1  # C01 gets +M5
        # M6 = (A01-A11) (B10+B11)
        U_in[1, 5] = 1; U_in[3, 5] = -1
        V_in[2, 5] = 1; V_in[3, 5] = 1
        W_in[3, 5] = 1  # C11 gets +M6
        # M7 = (A10-A00) (B00+B01)
        U_in[2, 6] = 1; U_in[0, 6] = -1
        V_in[0, 6] = 1; V_in[1, 6] = 1
        W_in[0, 6] = 1  # C00 gets +M7
        return U_in, V_in, W_in

    U_in, V_in, W_in = _inner_strassen()

    # Outer level: 4x4x4 split into 2x2x2 blocks of size 2x2x2.
    # A is 4x4, split into 4 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3).
    # B is 4x4 (for the first 4 cols), same block structure.
    # C is 4x4, same.
    #
    # Outer Strassen on the 2x2x2 block structure:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11) B00
    #   M3 = A00 (B01-B11)
    #   M4 = A11 (B10-B00)
    #   M5 = (A00+A10) B11
    #   M6 = (A01-A11) (B10+B11)
    #   M7 = (A10-A00) (B00+B01)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each outer product Mt, we need to compute a 2x2x2 matrix multiplication
    # (block A * block B = block C), which uses the inner Strassen (rank 7).
    # So total rank = 7 (outer) * 7 (inner) = 49.

    # Build the 4x4x4 Strassen decomposition:
    # U_outer: maps 4x4 A-blocks -> 7 outer products
    # V_outer: maps 4x4 B-blocks -> 7 outer products
    # W_outer: maps 4x4 C-blocks -> 7 outer products

    # Outer U: for each A element (4x4=16), coefficient in each of 7 outer products.
    # A block indices: A00 -> elements (0,0),(0,1),(1,0),(1,1) -> global (0,0),(0,1),(1,0),(1,1)
    # A01 -> elements (0,2),(0,3),(1,2),(1,3) -> global (0,2),(0,3),(1,2),(1,3)
    # A10 -> elements (2,0),(2,1),(3,0),(3,1) -> global (2,0),(2,1),(3,0),(3,1)
    # A11 -> elements (2,2),(2,3),(3,2),(3,3) -> global (2,2),(2,3),(3,2),(3,3)

    # Outer coefficients (same pattern as inner but at block level):
    # For A:
    # M1: A00+A11 -> blocks 0 and 3
    # M2: A10+A11 -> blocks 2 and 3
    # M3: A00 -> block 0
    # M4: A11 -> block 3
    # M5: A00+A10 -> blocks 0 and 2
    # M6: A01-A11 -> blocks 1 and 3
    # M7: A10-A00 -> blocks 2 and 0

    # We'll build U, V, W of shape (16, 49) for the 4x4x4 part.
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)

    # Block-to-global mapping for 2x2 blocks in a 4x4 matrix:
    # Block 0 (A00): rows 0-1, cols 0-1 -> elements [0,1,4,5] (row-major: i*4+j)
    # Block 1 (A01): rows 0-1, cols 2-3 -> elements [2,3,6,7]
    # Block 2 (A10): rows 2-3, cols 0-1 -> elements [8,9,12,13]
    # Block 3 (A11): rows 2-3, cols 2-3 -> elements [10,11,14,15]

    def block_elements(block_idx, m_dim):
        """Return global element indices for a 2x2 block in an m_dim x m_dim matrix."""
        br = block_idx // 2
        bc = block_idx % 2
        return [(br * 2 + r) * m_dim + (bc * 2 + c) for r in range(2) for c in range(2)]

    # Outer coefficients for A and B (same structure):
    outer_A_coeff = {
        0: {0: 1, 3: 1},   # M1: A00+A11
        1: {2: 1, 3: 1},   # M2: A10+A11
        2: {0: 1},          # M3: A00
        3: {3: 1},          # M4: A11
        4: {0: 1, 2: 1},   # M5: A00+A10
        5: {1: 1, 3: -1},  # M6: A01-A11
        6: {2: 1, 0: -1},  # M7: A10-A00
    }
    outer_B_coeff = {
        0: {0: 1, 3: 1},   # M1: B00+B11
        1: {0: 1},          # M2: B00
        2: {1: 1, 3: -1},  # M3: B01-B11
        3: {2: 1, 0: -1},  # M4: B10-B00
        4: {3: 1},          # M5: B11
        5: {2: 1, 3: 1},   # M6: B10+B11
        6: {0: 1, 1: 1},   # M7: B00+B01
    }
    outer_W_coeff = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},   # C00: M1+M4-M5+M7
        1: {2: 1, 4: 1},                # C01: M3+M5
        2: {1: 1, 3: 1},                # C10: M2+M4
        3: {0: 1, 1: -1, 2: 1, 5: 1},   # C11: M1-M2+M3+M6
    }

    # For each outer product t (0..6) and inner product tp (0..6):
    for t in range(7):
        for tp in range(7):
            col = t * 7 + tp
            # U: for each A element, coefficient = outer_A_coeff[t][block] * inner_U_in[elem_in_block, tp]
            for block in range(4):
                coeff = outer_A_coeff[t].get(block, 0)
                if coeff == 0:
                    continue
                elems = block_elements(block, 4)
                for local_idx, glob_idx in enumerate(elems):
                    U[glob_idx, col] += coeff * U_in[local_idx, tp]

            # V: for each B element
            for block in range(4):
                coeff = outer_B_coeff[t].get(block, 0)
                if coeff == 0:
                    continue
                elems = block_elements(block, 4)
                for local_idx, glob_idx in enumerate(elems):
                    V[glob_idx, col] += coeff * V_in[local_idx, tp]

            # W: for each C element
            for c_block in range(4):
                coeff = outer_W_coeff[c_block].get(t, 0)
                if coeff == 0:
                    continue
                elems = block_elements(c_block, 4)
                for local_idx, glob_idx in enumerate(elems):
                    W[glob_idx, col] += coeff * W_in[local_idx, tp]

    # Now append 16 extra rank-1 terms for the 5th column of B (k=4).
    # For each (i, j) pair, one term: U[i*4+j] = 1, V[j*4+4] = 1, W[4*4+i] = 1
    total_rank = 49 + 16
    U_full = np.zeros((16, total_rank), dtype=float)
    V_full = np.zeros((20, total_rank), dtype=float)
    W_full = np.zeros((20, total_rank), dtype=float)

    U_full[:, :49] = U
    V_full[:, :49] = V
    W_full[:, :49] = W

    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U_full[i * 4 + j, col] = 1.0
            V_full[j * 4 + 4, col] = 1.0
            W_full[4 * 4 + i, col] = 1.0

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
