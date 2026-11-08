import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    Uses 4 independent Strassen (rank-7) blocks for the 2x2 sub-products.
    """
    n, m, p = DIMENSIONS
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def add_term(u_dict, v_dict, w_dict, col_idx):
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx, col_idx] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx, col_idx] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx, col_idx] += coeff

    # Strassen terms for 2x2 matrix multiplication P = X * Y
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]

    def apply_strassen_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_local, v_local, w_local) in enumerate(strassen_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * 4 + (a_col_off * 2 + c)
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off * 2 + r) * 4 + (b_col_off * 2 + c)
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off * 2 + c) * 2 + r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            add_term(u_global, v_global, w_global, col)

    # C_L = A_L @ B_TL + A_R @ B_BL
    # C_R = A_L @ B_TR + A_R @ B_BR
    apply_strassen_block(0, 0, 0, 0, 0)   # A_L @ B_TL -> C_L
    apply_strassen_block(1, 1, 0, 0, 7)   # A_R @ B_BL -> C_L
    apply_strassen_block(0, 0, 1, 1, 14)  # A_L @ B_TR -> C_R
    apply_strassen_block(1, 1, 1, 1, 21)  # A_R @ B_BR -> C_R

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = U.shape[1]

    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-10):
            print(f"Test failed for trial {trial}")
            print(f"C:\n{C}")
            print(f"Cf:\n{Cf}")
            exit(1)
    print(f"OK rank {rank} in {time.time()-t0:.1f}s")
