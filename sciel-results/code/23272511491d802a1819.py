import numpy as np

def construct():
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7

    # Define Strassen 7-term decomposition as coefficient matrices
    # U: 4x7 (A indices: A00, A01, A10, A11)
    U_block = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    # V: 4x7 (B local indices: B00, B01, B10, B11)
    V_block = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, -1, 1, 1, 0, 0]
    ], dtype=int)

    # W: 4x7 (C local indices: C00, C01, C10, C11)
    W_block = np.array([
        [1, 0, 1, 0, -1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build full matrices using Kronecker-like block construction
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for q in range(num_blocks):
        r_slice = slice(q * 7, (q + 1) * 7)

        # U: A indices are global (0-3), same for all blocks
        U[0:4, r_slice] = U_block

        # V: B indices for block q
        # B is 2x16, flattened row-major: row * p + col
        # Block q uses columns 2q and 2q+1
        b_indices = [0 * p + 2 * q, 0 * p + (2 * q + 1), 1 * p + 2 * q, 1 * p + (2 * q + 1)]
        for li, gi in enumerate(b_indices):
            V[gi, r_slice] = V_block[li, :]

        # W: C indices for block q
        # C convention: k*n + i where k is column (0..15), i is row (0..1)
        # Block q uses columns 2q and 2q+1
        c_indices = [
            (2 * q) * n + 0, (2 * q + 1) * n + 0,
            (2 * q) * n + 1, (2 * q + 1) * n + 1
        ]
        for li, gi in enumerate(c_indices):
            W[gi, r_slice] = W_block[li, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
