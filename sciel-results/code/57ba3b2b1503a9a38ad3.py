"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Construct the rank-35 Strassen baseline efficiently.
The tensor for 2x2x10 matrix multiplication is the direct sum of five 2x2x2 tensors.
The rank of 2x2x2 is 7 (Strassen). Thus, the rank is 5 * 7 = 35.
This is the known minimal rank for this specific structure.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0


def _build_strassen_baseline():
    """Build the safe rank-35 Strassen baseline."""
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-multiplication decomposition for 2x2x2
    # Indices:
    # U: 0=A00, 1=A01, 2=A10, 3=A11
    # V: 0=B00, 1=B01, 2=B10, 3=B11
    # W: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Standard Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Representation: list of (index, coefficient) for each factor
    strassen_decomp = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            for v_koff, v_coeff in v_side:
                # v_koff is 0..3 for B00, B01, B10, B11
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            for w_koff, w_coeff in w_side:
                # w_koff is 0..3 for C00, C01, C10, C11
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    return U, V, W, rank


def _verify_decomposition(U, V, W, rank, n, m, p):
    """Verify that the decomposition exactly performs matrix multiplication."""
    rng = np.random.default_rng(12345)
    A_test = rng.integers(-5, 6, size=(n, m))
    B_test = rng.integers(-5, 6, size=(m, p))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_computed = np.zeros(p * n)
    for r in range(rank):
        uA = np.dot(U_arr[:, r], A_flat)
        vB = np.dot(V_arr[:, r], B_flat)
        C_computed += uA * vB * W_arr[:, r]
    
    C_expected_reorder = np.zeros(p * n)
    for i in range(n):
        for k in range(p):
            C_expected_reorder[k * n + i] = C_expected[i, k]
    
    return np.allclose(C_computed, C_expected_reorder, atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Construct the rank-35 Strassen baseline
    U, V, W, rank = _build_strassen_baseline()
    
    # Verify the decomposition to ensure correctness
    if not _verify_decomposition(U, V, W, rank, n, m, p):
        # Fallback: Schoolbook rank 40 (though this should never happen)
        # This is a safety net, not expected to be used.
        U_sb = [[0] * 40 for _ in range(n * m)]
        V_sb = [[0] * 40 for _ in range(m * p)]
        W_sb = [[0] * 40 for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][idx] = 1
                    V_sb[j * p + k][idx] = 1
                    W_sb[k * n + i][idx] = 1
                    idx += 1
        U, V, W, rank = U_sb, V_sb, W_sb, 40
        if not _verify_decomposition(U, V, W, rank, n, m, p):
            raise RuntimeError("Decomposition verification failed")
            
    return {"U": U, "V": V, "W": W}
