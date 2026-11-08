import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS

    # Use a direct, non-recursive Kronecker-product construction for 3x4x6.
    # Split p = 6 into three blocks of size 2.
    # For each p-block, split m = 4 into two blocks of size 2.
    # For each (m-block, p-block), split n = 3 into n=2 (Strassen) and n=1 (schoolbook).
    # This yields 3 * 2 * (7 + 4) = 66 rank terms, but we generate them
    # via explicit Kronecker products of the base 2x2x2 and 1x2x2 tensors.

    # Base 2x2x2 Strassen tensor (rank 7)
    # U, V, W are lists of (indices, values) tuples for each rank-1 term
    S_U = [
        [(0, 0), (1, 1)],
        [(1, 0), (1, 1)],
        [(0, 0)],
        [(1, 1)],
        [(0, 0), (0, 1)],
        [(1, 0), (0, 0)],
        [(0, 1), (1, 1)],
    ]
    S_V = [
        [(0, 0), (1, 1)],
        [(0, 0)],
        [(0, 1), (1, 1)],
        [(1, 0), (0, 0)],
        [(1, 1)],
        [(0, 0), (0, 1)],
        [(1, 0), (1, 1)],
    ]
    S_W = [
        [(0, 0), (1, 1)],
        [(1, 0), (1, 1)],
        [(0, 1), (1, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
        [(1, 1)],
        [(0, 0)],
    ]
    S_Uv = [[1, 1], [1, 1], [1], [1], [1, 1], [1, -1], [1, -1]]
    S_Vv = [[1, 1], [1], [1, -1], [1, -1], [1], [1, 1], [1, 1]]
    S_Wv = [[1, 1], [1, -1], [1, 1], [1, 1], [-1, 1], [1], [1]]

    # Base 1x2x2 schoolbook tensor (rank 4)
    B_U = [
        [(0, 0)],
        [(0, 1)],
        [(1, 0)],
        [(1, 1)],
    ]
    B_V = [
        [(0, 0)],
        [(0, 1)],
        [(1, 0)],
        [(1, 1)],
    ]
    B_W = [
        [(0, 0)],
        [(0, 1)],
        [(1, 0)],
        [(1, 1)],
    ]
    B_Uv = [[1], [1], [1], [1]]
    B_Vv = [[1], [1], [1], [1]]
    B_Wv = [[1], [1], [1], [1]]

    total_rank = 3 * 2 * (7 + 4)  # 66
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0

    # Iterate over p-blocks (3 blocks of size 2)
    for pb in range(3):
        # Iterate over m-blocks (2 blocks of size 2)
        for mb in range(2):
            # --- 2x2x2 Strassen part ---
            for t in range(7):
                u_idx = []
                u_val = []
                v_idx = []
                v_val = []
                w_idx = []
                w_val = []

                for (ri, rj), val in zip(S_U[t], S_Uv[t]):
                    # n-index: 0 or 1, m-index: 2*mb + rj, p-index: 2*pb + rk (not used in U)
                    ui = ri * m + (2 * mb + rj)
                    u_idx.append(ui)
                    u_val.append(val)

                for (rj, rk), val in zip(S_V[t], S_Vv[t]):
                    # m-index: 2*mb + rj, p-index: 2*pb + rk
                    vi = (2 * mb + rj) * p + (2 * pb + rk)
                    v_idx.append(vi)
                    v_val.append(val)

                for (rk, ri), val in zip(S_W[t], S_Wv[t]):
                    # p-index: 2*pb + rk, n-index: ri
                    wi = (2 * pb + rk) * n + ri
                    w_idx.append(wi)
                    w_val.append(val)

                for idx, val in zip(u_idx, u_val):
                    U[idx, col] += val
                for idx, val in zip(v_idx, v_val):
                    V[idx, col] += val
                for idx, val in zip(w_idx, w_val):
                    W[idx, col] += val
                col += 1

            # --- 1x2x2 schoolbook part (for n-index 2) ---
            for t in range(4):
                u_idx = []
                u_val = []
                v_idx = []
                v_val = []
                w_idx = []
                w_val = []

                for (ri, rj), val in zip(B_U[t], B_Uv[t]):
                    # n-index: 2 + ri (but ri is always 0 in B_U, so n-index = 2)
                    # Actually B_U has indices (0,0),(0,1),(1,0),(1,1) but n=1 so ri is always 0
                    # Wait, B_U is for 1x2x2, so indices are (j) for U, (j,k) for V, (k) for W
                    # Let me re-examine: for 1x2x2, U is (1*2, rank), V is (2*2, rank), W is (2*1, rank)
                    # U index: j (0 or 1), V index: (j, k), W index: k (0 or 1)
                    # In the full tensor, n-index = 2, m-index = 2*mb + j, p-index = 2*pb + k
                    ui = 2 * m + (2 * mb + rj)
                    u_idx.append(ui)
                    u_val.append(val)

                for (rj, rk), val in zip(B_V[t], B_Vv[t]):
                    vi = (2 * mb + rj) * p + (2 * pb + rk)
                    v_idx.append(vi)
                    v_val.append(val)

                for (rk, ri), val in zip(B_W[t], B_Wv[t]):
                    wi = (2 * pb + rk) * n + 2  # n-index = 2
                    w_idx.append(wi)
                    w_val.append(val)

                for idx, val in zip(u_idx, u_val):
                    U[idx, col] += val
                for idx, val in zip(v_idx, v_val):
                    V[idx, col] += val
                for idx, val in zip(w_idx, w_val):
                    W[idx, col] += val
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
