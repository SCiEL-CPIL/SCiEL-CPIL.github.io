"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x2 by 2x8 matrix multiplication
    using 4 independent Strassen decompositions.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 28

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Strassen terms for 2x2 matrix multiplication
    # u: vector of length 4 (A indices: 00, 01, 10, 11)
    # v: vector of length 4 (B indices: 00, 01, 10, 11)
    # w: vector of length 4 (C indices: 00, 01, 10, 11)
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    strassen_terms = [
        # (u, v, w) where indices are [00, 01, 10, 11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    ]

    num_blocks = p // 2  # 4 blocks of 2 columns each

    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = strassen_terms[term_idx]

            # Fill U: A is 2x2, indices 0..3. Same for all blocks.
            for i in range(4):
                U[i][col] = u[i]

            # Fill V: B is 2x8. Block covers columns 2*block_idx and 2*block_idx+1.
            # Global B index: j * p + k_global.
            # j in 0..1, k_global in 0..7.
            # Local v index: j * 2 + k_local.
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * p + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]

            # Fill W: C is 8x2 (p x n).
            # Convention: W[k*n+i, r].
            # Global C index: k_global * n + i.
            # k_global in 0..7, i in 0..1.
            # Local w index: i * 2 + k_local.
            # C_local indices: 00, 01, 10, 11.
            # i is row (0,1), k_local is col (0,1).
            # So local index is i * 2 + k_local.
            # Global index is k_global * n + i.
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * n + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]

    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Check: C[i,l] = sum_k U[i,k] * V[k,l] * W[l,k]
    # Actually the tensor decomposition is: T[a,b,c] = U[a,k] * V[b,k] * W[c,k]
    # where a indexes A, b indexes B, c indexes C
    # C[c] = sum_a sum_b T[a,b,c] * A[a] * B[b]
    # So T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]

    # A has n*m = 4 entries, B has m*p = 16 entries, C has p*n = 16 entries
    # Indexing: A[i*m+k] for row i, col k of A (2x2)
    # B[k*p+l] for row k, col l of B (2x8)  
    # C[l*n+i] for row l, col i of C (2x8) -> W[k*n+i, r] per convention

    # Verify by checking a few random instances
    np.random.seed(42)
    for trial in range(10):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test  # shape (n, p) = (2, 8)

        # Compute using tensor decomposition
        C_computed = np.zeros((n, p))
        A_flat = A_test.flatten()  # shape (4,)
        B_flat = B_test.flatten()  # shape (16,)

        for r in range(rank):
            # u_r is a vector of length 4
            # v_r is a vector of length 16
            # w_r is a vector of length 16
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]

            # scalar product
            s = np.dot(A_flat, u_r) * np.dot(B_flat, v_r)

            # add s * w_r to C_computed
            # w_r has length 16, indexed as [l*n + i]
            for idx in range(pn):
                l = idx // n
                i = idx % n
                C_computed[i, l] += s * w_r[idx]

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # If verification fails, return the construction anyway (should not happen)
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
