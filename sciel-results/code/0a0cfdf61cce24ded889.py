import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140.0

    # Construct rank-56 Strassen decomposition.
    rank_56 = 56
    U56 = np.zeros((8, rank_56))
    V56 = np.zeros((32, rank_56))
    W56 = np.zeros((16, rank_56))

    col = 0
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6),
        (2, 0, 0),
        (2, 0, 2),
        (2, 0, 4),
        (2, 0, 6),
    ]

    for (a_row_off, b_row_off, b_col_off) in blocks:
        a00 = a_row_off + 0
        a01 = a_row_off + 1
        a10 = a_row_off + 4 + 0
        a11 = a_row_off + 4 + 1

        b00 = b_row_off * 8 + b_col_off
        b01 = b_row_off * 8 + b_col_off + 1
        b10 = (b_row_off + 1) * 8 + b_col_off
        b11 = (b_row_off + 1) * 8 + b_col_off + 1

        c00 = b_col_off * 2 + 0
        c01 = b_col_off * 2 + 1
        c10 = (b_col_off + 1) * 2 + 0
        c11 = (b_col_off + 1) * 2 + 1

        # Strassen terms
        U56[a00, col] += 1; U56[a11, col] += 1
        V56[b00, col] += 1; V56[b11, col] += 1
        W56[c00, col] += 1; W56[c11, col] += 1
        col += 1

        U56[a10, col] += 1; U56[a11, col] += 1
        V56[b00, col] += 1
        W56[c10, col] += 1; W56[c11, col] += -1
        col += 1

        U56[a00, col] += 1
        V56[b01, col] += 1; V56[b11, col] += -1
        W56[c01, col] += 1; W56[c11, col] += 1
        col += 1

        U56[a11, col] += 1
        V56[b10, col] += 1; V56[b00, col] += -1
        W56[c00, col] += 1; W56[c10, col] += 1
        col += 1

        U56[a00, col] += 1; U56[a01, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += -1; W56[c01, col] += 1
        col += 1

        U56[a10, col] += 1; U56[a00, col] += -1
        V56[b00, col] += 1; V56[b01, col] += 1
        W56[c11, col] += 1
        col += 1

        U56[a01, col] += 1; U56[a11, col] += -1
        V56[b10, col] += 1; V56[b11, col] += 1
        W56[c00, col] += 1
        col += 1

    assert col == 56

    # Build tensor matrix and check for linear dependencies.
    M = np.zeros((64, 56))
    for r in range(56):
        for i in range(2):
            for a in range(4):
                for c in range(8):
                    M[i * 32 + a * 8 + c, r] = U56[i * 4 + a, r] * V56[a * 8 + c, r] * W56[c * 2 + i, r]

    rank_M = np.linalg.matrix_rank(M, tol=1e-9)

    best_U, best_V, best_W = U56.copy(), V56.copy(), W56.copy()
    best_rank = 56

    # Try to remove linearly dependent terms.
    if time.time() < deadline - 10:
        removed = []
        for k in range(56):
            if time.time() > deadline - 5:
                break
            if k in removed:
                continue
            # Check if term k is in the span of the other terms.
            other_indices = [i for i in range(56) if i != k and i not in removed]
            M_other = M[:, other_indices]
            rank_other = np.linalg.matrix_rank(M_other, tol=1e-9)
            if rank_other == rank_M:
                # Term k is linearly dependent.
                removed.append(k)

        if removed:
            keep_indices = [i for i in range(56) if i not in removed]
            best_U = U56[:, keep_indices]
            best_V = V56[:, keep_indices]
            best_W = W56[:, keep_indices]
            best_rank = len(keep_indices)

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
