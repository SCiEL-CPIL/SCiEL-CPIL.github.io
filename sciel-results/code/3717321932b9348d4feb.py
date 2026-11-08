import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2x2 terms: (u_local, v_local, w_local)
    # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Standard Strassen for P = X*Y (2x2):
    #   M1 = (X00+X11)(Y00+Y11)
    #   M2 = (X10+X11)Y00
    #   M3 = X00(Y01-Y11)
    #   M4 = X11(Y10-Y00)
    #   M5 = (X00+X01)Y11
    #   M6 = (X10-X00)(Y00+Y01)
    #   M7 = (X01-X11)(Y10+Y11)
    #   P00 = M1 + M4 - M5 + M7
    #   P01 = M3 + M5
    #   P10 = M2 + M4
    #   P11 = M1 - M2 + M3 + M6
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        """Apply Strassen terms to a specific 2x2x2 block.
        
        a_col_off: column offset in A (0 or 2)
        b_row_off: row offset in B (0 or 2)
        b_col_off: column offset in B (0 or 2)
        c_col_off: column offset in C (0 or 2)
        start_col: starting column in the factor matrices
        """
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off + r) * p + (b_col_off + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                # W[k*n + i] where k=column of C, i=row of C
                g_idx = (c_col_off + c) * n + r
                W[g_idx, col] += coeff

    # Four independent 2x2x2 Strassen blocks:
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    apply_strassen(0, 0, 0, 0, 0)
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    apply_strassen(2, 2, 0, 0, 7)
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    apply_strassen(0, 0, 2, 2, 14)
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    apply_strassen(2, 2, 2, 2, 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    # Test with multiple random matrices
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B

        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc

        if not np.allclose(C, Cf, atol=1e-10):
            print(f"Mismatch seed={seed}")
            print("C:", C)
            print("Cf:", Cf)
            break
    else:
        print("OK rank", U.shape[1])
