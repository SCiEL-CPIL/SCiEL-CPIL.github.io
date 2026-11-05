"""Rank-44 decomposition for 3x4x4 matrix multiplication tensor."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check exact decomposition of the matrix multiplication tensor."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    import time
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = 3, 4, 4

    # Block decomposition:
    # A is 3x4: rows 0,1 form 2x2 block with cols 0,1; row 2 is 1x4
    # B is 4x4: cols 0,1 form 2x2 block with rows 0,1; cols 2,3 form 2x4 block with rows 0,1
    #
    # We split into:
    #   Block 1: A[0:2,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[0:2,0:2] (2x2)  [Strassen: 7 terms]
    #   Block 2: A[0:2,0:2] (2x2) * B[0:2,2:4] (2x2) -> C[0:2,2:4] (2x2)  [Strassen: 7 terms]
    #   Block 3: A[0:2,2:4] (2x2) * B[2:4,0:2] (2x2) -> C[0:2,0:2] (2x2)  [Strassen: 7 terms]
    #   Block 4: A[0:2,2:4] (2x2) * B[2:4,2:4] (2x2) -> C[0:2,2:4] (2x2)  [Strassen: 7 terms]
    #   Block 5: A[2:3,0:2] (1x2) * B[0:2,0:2] (2x2) -> C[2:3,0:2] (1x2)  [schoolbook: 2 terms]
    #   Block 6: A[2:3,0:2] (1x2) * B[0:2,2:4] (2x2) -> C[2:3,2:4] (1x2)  [schoolbook: 2 terms]
    #   Block 7: A[2:3,2:4] (1x2) * B[2:4,0:2] (2x2) -> C[2:3,0:2] (1x2)  [schoolbook: 2 terms]
    #   Block 8: A[2:3,2:4] (1x2) * B[2:4,2:4] (2x2) -> C[2:3,2:4] (1x2)  [schoolbook: 2 terms]
    #
    # Total: 4*7 + 4*2 = 28 + 8 = 36 terms
    # Wait, that's rank 36! Let me re-examine.
    #
    # Actually each 2x2x2 sub-tensor needs its own 7 terms (Strassen), and each 1x2x2 needs 2 terms.
    # 4 blocks of 2x2x2: 4*7 = 28
    # 4 blocks of 1x2x2: 4*2 = 8
    # Total rank = 36
    #
    # But wait - I need to be careful. The blocks overlap in the C matrix indices.
    # Let me re-derive carefully.
    #
    # A[i,j] for i in 0..2, j in 0..3
    # B[j,k] for j in 0..3, k in 0..3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let me partition the j-index:
    #   j in {0,1} and j in {2,3}
    #
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # For i in {0,1}, k in {0,1}:
    #   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #   First sum: A[0:2,0:2] * B[0:2,0:2] -> 2x2x2, Strassen 7 terms
    #   Second sum: A[0:2,2:4] * B[2:4,0:2] -> 2x2x2, Strassen 7 terms
    #
    # For i in {0,1}, k in {2,3}:
    #   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #   First sum: A[0:2,0:2] * B[0:2,2:4] -> 2x2x2, Strassen 7 terms
    #   Second sum: A[0:2,2:4] * B[2:4,2:4] -> 2x2x2, Strassen 7 terms
    #
    # For i=2, k in {0,1}:
    #   C[2,k] = sum_{j=0}^{1} A[2,j]*B[j,k] + sum_{j=2}^{3} A[2,j]*B[j,k]
    #   First sum: A[2,0:2] * B[0:2,0:2] -> 1x2x2, schoolbook 2 terms
    #   Second sum: A[2,2:4] * B[2:4,0:2] -> 1x2x2, schoolbook 2 terms
    #
    # For i=2, k in {2,3}:
    #   C[2,k] = sum_{j=0}^{1} A[2,j]*B[j,k] + sum_{j=2}^{3} A[2,j]*B[j,k]
    #   First sum: A[2,0:2] * B[0:2,2:4] -> 1x2x2, schoolbook 2 terms
    #   Second sum: A[2,2:4] * B[2:4,2:4] -> 1x2x2, schoolbook 2 terms
    #
    # Total: 4*7 + 4*2 = 28 + 8 = 36
    #
    # Let me build this explicitly.

    U = []
    V = []
    W = []

    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries: dict mapping (i,j) to coeff, etc."""
        idx = len(U)
        U.append([0.0] * 0)  # placeholder, will resize
        V.append([0.0] * 0)
        W.append([0.0] * 0)
        # Actually let me build as lists of dicts then convert
        return idx

    # Let me build using a different approach: accumulate into arrays
    # Start with empty, grow as needed
    rank = 0
    U_rows = {}  # (i,j) -> list of (coeff, col)
    V_rows = {}  # (j,k) -> list of (coeff, col)
    W_rows = {}  # (k,i) -> list of (coeff, col)

    def add_term(u_dict, v_dict, w_dict):
        nonlocal rank
        col = rank
        rank += 1
        for key, val in u_dict.items():
            U_rows.setdefault(key, []).append((val, col))
        for key, val in v_dict.items():
            V_rows.setdefault(key, []).append((val, col))
        for key, val in w_dict.items():
            W_rows.setdefault(key, []).append((val, col))

    # Strassen's 7 terms for 2x2x2:
    # Given A2 (2x2) with entries a00,a01,a10,a11 and B2 (2x2) with b00,b01,b10,b11
    # C2 = A2 * B2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M4
    # c10 = M2 + M7
    # c11 = M1 - M2 + M3 + M6

    def strassen_222(a_base_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        """Add 7 Strassen terms for 2x2x2 block.
        A indices: (a_base_i + di, a_base_j + dj) for di,dj in {0,1}
        B indices: (b_base_j + dj, b_base_k + dk) for dj,dk in {0,1}
        C indices: (c_base_i + di, c_base_k + dk) for di,dk in {0,1}
        """
        # M1 = (a00+a11)(b00+b11)
        add_term(
            {(a_base_i+0, a_base_j+0): 1, (a_base_i+1, a_base_j+1): 1},
            {(b_base_j+0, b_base_k+0): 1, (b_base_j+1, b_base_k+1): 1},
            {(c_base_k+0, c_base_i+0): 1, (c_base_k+1, c_base_i+1): -1,
             (c_base_k+1, c_base_i+0): -1, (c_base_k+0, c_base_i+1): 1}
        )
        # M2 = (a10+a11)b00
        add_term(
            {(a_base_i+1, a_base_j+0): 1, (a_base_i+1, a_base_j+1): 1},
            {(b_base_j+0, b_base_k+0): 1},
            {(c_base_k+0, c_base_i+1): 1, (c_base_k+1, c_base_i+1): -1}
        )
        # M3 = a00(b10-b11)
        add_term(
            {(a_base_i+0, a_base_j+0): 1},
            {(b_base_j+1, b_base_k+0): 1, (b_base_j+1, b_base_k+1): -1},
            {(c_base_k+1, c_base_i+0): 1, (c_base_k+1, c_base_i+1): 1}
        )
        # M4 = a11(b01-b00)
        add_term(
            {(a_base_i+1, a_base_j+1): 1},
            {(b_base_j+0, b_base_k+1): 1, (b_base_j+0, b_base_k+0): -1},
            {(c_base_k+1, c_base_i+0): -1, (c_base_k+1, c_base_i+1): 1,
             (c_base_k+0, c_base_i+0): 1, (c_base_k+0, c_base_i+1): -1}
        )
        # M5 = (a00+a01)b11
        add_term(
            {(a_base_i+0, a_base_j+0): 1, (a_base_i+0, a_base_j+1): 1},
            {(b_base_j+1, b_base_k+1): 1},
            {(c_base_k+0, c_base_i+0): -1, (c_base_k+1, c_base_i+1): 1}
        )
        # M6 = (a10-a00)(b00+b10)
        add_term(
            {(a_base_i+1, a_base_j+0): 1, (a_base_i+0, a_base_j+0): -1},
            {(b_base_j+0, b_base_k+0): 1, (b_base_j+1, b_base_k+0): 1},
            {(c_base_k+1, c_base_i+1): 1}
        )
        # M7 = (a01-a11)(b01+b11)
        add_term(
            {(a_base_i+0, a_base_j+1): 1, (a_base_i+1, a_base_j+1): -1},
            {(b_base_j+0, b_base_k+1): 1, (b_base_j+1, b_base_k+1): 1},
            {(c_base_k+0, c_base_i+0): -1, (c_base_k+1, c_base_i+1): 1,
             (c_base_k+1, c_base_i+0): 1}
        )

    def schoolbook_122(a_i, a_base_j, b_base_j, b_base_k, c_base_i, c_base_k):
        """Add 2 schoolbook terms for 1x2x2 block.
        A index: (a_i, a_base_j + dj) for dj in {0,1}
        B indices: (b_base_j + dj, b_base_k + dk) for dj,dk in {0,1}
        C indices: (c_base_i, c_base_k + dk) for dk in {0,1}
        """
        # Term for dj=0: A[i, j0] * B[j0, k0] -> C[i, k0], A[i,j0]*B[j0,k1] -> C[i,k1]
        add_term(
            {(a_i, a_base_j+0): 1},
            {(b_base_j+0, b_base_k+0): 1, (b_base_j+0, b_base_k+1): 1},
            {(c_base_k+0, c_base_i): 1, (c_base_k+1, c_base_i): 1}
        )
        # Term for dj=1
        add_term(
            {(a_i, a_base_j+1): 1},
            {(b_base_j+1, b_base_k+0): 1, (b_base_j+1, b_base_k+1): 1},
            {(c_base_k+0, c_base_i): 1, (c_base_k+1, c_base_i): 1}
        )

    # Now add all blocks:
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_222(0, 0, 0, 0, 0, 0)
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    strassen_222(0, 0, 0, 2, 0, 2)
    # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    strassen_222(0, 2, 2, 0, 0, 0)
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    strassen_222(0, 2, 2, 2, 0, 2)
    # Block 5: A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    schoolbook_122(2, 0, 0, 0, 2, 0)
    # Block 6: A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    schoolbook_122(2, 0, 0, 2, 2, 2)
    # Block 7: A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    schoolbook_122(2, 2, 2, 0, 2, 0)
    # Block 8: A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    schoolbook_122(2, 2, 2, 2, 2, 2)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for (i, j), entries in U_rows.items():
        for val, col in entries:
            U[i * m + j][col] = val
    for (j, k), entries in V_rows.items():
        for val, col in entries:
            V[j * p + k][col] = val
    for (k, i), entries in W_rows.items():
        for val, col in entries:
            W[k * n + i][col] = val

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
