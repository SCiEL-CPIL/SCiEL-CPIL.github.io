"""Matrix multiplication 2x4x5 - rank-36 Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact rank-36 tensor decomposition for 2x4 x 4x5 matrix multiplication.
    Uses Strassen (rank 7) for each 2x2x2 block and schoolbook (rank 4) for each 2x2x1 block.
    Total: 2 blocks * (2*7 + 4) = 2 * 18 = 36.
    """
    n, m, p = DIMENSIONS  # n=2 rows, m=4 inner, p=5 cols
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        for (i, j), val in u_coeffs.items():
            U[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i, idx] += val

    def strassen_block(a_col_off, b_row_off, c_col_off, idx_start):
        """
        Strassen decomposition for A[0:2, a_col_off:a_col_off+2] * B[b_row_off:b_row_off+2, c_col_off:c_col_off+2].
        Local indices: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]].
        P1 = (a+d)(e+h)
        P2 = (c+d)e
        P3 = a(f-h)
        P4 = d(g-e)
        P5 = (a+b)h
        P6 = (c-a)(e+f)
        P7 = (b-d)(g+h)
        C00 = P1 + P4 - P5 + P7
        C01 = P3 + P5
        C10 = P2 + P4
        C11 = P1 - P2 + P3 + P6
        """
        ac = a_col_off      # A column offset
        ar0, ar1 = 0, 1     # A row indices (local)
        br = b_row_off      # B row offset
        bc = c_col_off      # C column offset
        br0, br1 = br, br + 1
        bc0, bc1 = bc, bc + 1

        # P1 = (a+d)(e+h)
        add_term(idx_start,
                 {(ar0, ac): 1, (ar1, ac + 1): 1},
                 {(br0, bc0): 1, (br1, bc1): 1},
                 {(bc0, 0): 1, (bc1, 1): 1})

        # P2 = (c+d)e
        add_term(idx_start + 1,
                 {(ar1, ac): 1, (ar1, ac + 1): 1},
                 {(br0, bc0): 1},
                 {(bc0, 1): 1, (bc1, 1): -1})

        # P3 = a(f-h)
        add_term(idx_start + 2,
                 {(ar0, ac): 1},
                 {(br0, bc1): 1, (br1, bc1): -1},
                 {(bc1, 0): 1, (bc1, 1): 1})

        # P4 = d(g-e)
        add_term(idx_start + 3,
                 {(ar1, ac + 1): 1},
                 {(br1, bc0): 1, (br0, bc0): -1},
                 {(bc0, 0): 1, (bc0, 1): 1})

        # P5 = (a+b)h
        add_term(idx_start + 4,
                 {(ar0, ac): 1, (ar0, ac + 1): 1},
                 {(br1, bc1): 1},
                 {(bc0, 0): -1, (bc1, 0): 1})

        # P6 = (c-a)(e+f)
        add_term(idx_start + 5,
                 {(ar1, ac): 1, (ar0, ac): -1},
                 {(br0, bc0): 1, (br0, bc1): 1},
                 {(bc1, 1): 1})

        # P7 = (b-d)(g+h)
        add_term(idx_start + 6,
                 {(ar0, ac + 1): 1, (ar1, ac + 1): -1},
                 {(br1, bc0): 1, (br1, bc1): 1},
                 {(bc0, 0): 1})

    def schoolbook_2x2x1(a_col_off, b_row_off, c_col_off, idx_start):
        """
        Schoolbook for A[0:2, a_col_off:a_col_off+2] * B[b_row_off:b_row_off+2, c_col_off:c_col_off+1].
        C[0,0] = a*e + b*g
        C[1,0] = c*e + d*g
        """
        ac = a_col_off
        br = b_row_off
        bc = c_col_off
        br0, br1 = br, br + 1

        # a * e -> C[0, bc]
        add_term(idx_start,
                 {(0, ac): 1},
                 {(br0, bc): 1},
                 {(bc, 0): 1})

        # b * g -> C[0, bc]
        add_term(idx_start + 1,
                 {(0, ac + 1): 1},
                 {(br1, bc): 1},
                 {(bc, 0): 1})

        # c * e -> C[1, bc]
        add_term(idx_start + 2,
                 {(1, ac): 1},
                 {(br0, bc): 1},
                 {(bc, 1): 1})

        # d * g -> C[1, bc]
        add_term(idx_start + 3,
                 {(1, ac + 1): 1},
                 {(br1, bc): 1},
                 {(bc, 1): 1})

    idx = 0
    # Block 1: A[:, 0:2] * B[0:2, :]
    strassen_block(0, 0, 0, idx)
    idx += 7
    strassen_block(0, 0, 2, idx)
    idx += 7
    schoolbook_2x2x1(0, 0, 4, idx)
    idx += 4

    # Block 2: A[:, 2:4] * B[2:4, :]
    strassen_block(2, 2, 0, idx)
    idx += 7
    strassen_block(2, 2, 2, idx)
    idx += 7
    schoolbook_2x2x1(2, 2, 4, idx)
    idx += 4

    assert idx == rank, f"Rank mismatch: expected {rank}, got {idx}"

    # Verify correctness
    U_np = U
    V_np = V
    W_np = W
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if k is the column and i is the row, j is the inner index
    # Actually the matrix multiplication tensor: T[i,j,k] = delta_{i,i'} * ... 
    # For matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor: T[i,j,k] = 1 (it's the identity in the j-index sense, but really it's the structure tensor)
    # The structure tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition says: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k

    # Let's verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at i={i}, j={j}, k={k}: got {val}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
