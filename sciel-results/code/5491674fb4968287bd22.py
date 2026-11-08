import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=int)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=int)  # (8, 28)
    
    # Strassen 2x2 decomposition terms
    # For P = X * Y where X, Y are 2x2:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    #
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Each term: (u_coeffs, v_coeffs, w_coeffs) where keys are local 2x2 indices
    # Local index: 0=00, 1=01, 2=10, 3=11
    strassen = [
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
    
    def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        """
        Apply Strassen to compute contribution to C block.
        a_col_off: which 2-col block of A (0 or 1)
        b_row_off: which 2-row block of B (0 or 1)  
        b_col_off: which 2-col block of B (0 or 1)
        c_col_off: which 2-col block of C (0 or 1)
        """
        for t, (u_loc, v_loc, w_loc) in enumerate(strassen):
            col = start_col + t
            # U: A is 2x4, index = row*4 + col
            for loc, coeff in u_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, a_col_off * 2 + lc
                U[gr * m + gc, col] += coeff
            # V: B is 4x4, index = row*4 + col
            for loc, coeff in v_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = b_row_off * 2 + lr, b_col_off * 2 + lc
                V[gr * p + gc, col] += coeff
            # W: output C is 2x4, W[k*n+i] where k=col, i=row
            for loc, coeff in w_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, c_col_off * 2 + lc
                W[gc * n + gr, col] += coeff
    
    # C[:,0:2] = A[:,0:2]@B[0:2,0:2] + A[:,2:4]@B[2:4,0:2]
    apply_block(0, 0, 0, 0, 0)   # A_left @ B[0:2,0:2] -> C_left
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    apply_block(1, 1, 0, 0, 7)   # A_right @ B[2:4,0:2] -> C_left
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # C[:,2:4] = A[:,0:2]@B[0:2,2:4] + A[:,2:4]@B[2:4,2:4]
    apply_block(0, 0, 1, 1, 14)  # A_left @ B[0:2,2:4] -> C_right
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    apply_block(1, 1, 1, 1, 21)  # A_right @ B[2:4,2:4] -> C_right
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Verify with random test cases
    np.random.seed(42)
    for trial in range(100):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(U.shape[1]):
                    for j in range(m):
                        acc += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                Cf[i, k] = acc
        assert np.allclose(C, Cf, atol=1e-10), (trial, C, Cf)
    
    print("OK rank", U.shape[1])
