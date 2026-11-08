import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Try to find a rank-24 decomposition using a smarter blocking
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A is 2x4, B is 4x4, C is 2x4
    # W index: W[k*n + i, r] = W[k*2 + i, r]
    
    # Strategy: Use rank 24 by applying a 4-term Strassen-like decomposition
    # to each pair of columns of B, but share some factors.
    #
    # Actually, let's try a direct construction with rank 21.
    # We'll use the following structure:
    # Split B into 2 row-blocks of 2x4 each: B_top (rows 0,1), B_bot (rows 2,3)
    # C = A[:,0:2] @ B_top + A[:,2:4] @ B_bot
    # Each term is (2x2) @ (2x4) -> 2x4
    # A 2x2 @ 2x4 can be done with rank 9 (known result for 2x2x4 tensor)
    # But we have two such terms, so rank 18? No, they share A factors.
    #
    # Let me try a different approach: direct rank search with structured ansatz.
    #
    # For now, let's use a known rank-24 decomposition:
    # Apply Strassen to compute C = A @ B by viewing it as:
    # C[:, 0:2] = (A[:,0:2] + A[:,2:4]) @ (B[0:2,0:2] + B[2:4,0:2]) 
    #            - A[:,2:4] @ B[0:2,0:2] - A[:,0:2] @ B[2:4,0:2]
    # This doesn't obviously reduce rank.
    #
    # Let me just use the rank-28 solution but verify it's correct,
    # and try to see if we can shave off a few ranks.
    
    # Use the proven rank-28 decomposition
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=int)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=int)  # (8, 28)
    
    # Strassen 2x2 decomposition terms
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t, (u_loc, v_loc, w_loc) in enumerate(strassen):
            col = start_col + t
            for loc, coeff in u_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, a_col_off * 2 + lc
                U[gr * m + gc, col] += coeff
            for loc, coeff in v_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = b_row_off * 2 + lr, b_col_off * 2 + lc
                V[gr * p + gc, col] += coeff
            for loc, coeff in w_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, c_col_off * 2 + lc
                W[gc * n + gr, col] += coeff
    
    apply_block(0, 0, 0, 0, 0)
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    apply_block(1, 1, 0, 0, 7)
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    apply_block(0, 0, 1, 1, 14)
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    apply_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
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
