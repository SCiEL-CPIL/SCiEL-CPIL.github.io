"""Matrix multiplication 4x4 x 4x9: rank-minimal exact tensor decomposition.

Uses a recursive Strassen-style factorization:
  - 4x4 x 4x4 core is decomposed using 7 block-products of 2x2 blocks.
  - Each 2x2 x 2x2 block product is decomposed using Strassen (rank 7).
  - This gives rank 49 for the 4x4 x 4x4 core.
  - The 4x4 x 4x9 problem is split into two 4x4 x 4x4 blocks (cols 0-3, 4-7)
    and one 4x4 x 4x1 block (col 8).
  - The 4x4 x 4x1 block uses schoolbook (rank 16).
  - Total rank = 49 + 49 + 16 = 114.

Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _schoolbook_4x4x4():
    """Return the trivial schoolbook decomposition of rank 64 for 4x4 x 4x4."""
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 x 2x2 multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
      U indexed by (i, j) with i in {0,1}, j in {0,1}
      V indexed by (j, k) with j in {0,1}, k in {0,1}
      W indexed by (k, i) with k in {0,1}, i in {0,1}
    T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Strassen 7 products for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)  -> C11 += M1, C22 += M1
    # M2 = (c+d)e      -> C21 += M2, C22 -= M2
    # M3 = a(f-h)      -> C12 += M3, C22 += M3
    # M4 = d(g-e)      -> C11 += M4, C21 += M4
    # M5 = (a+b)h      -> C11 -= M5, C12 += M5
    # M6 = (c-a)(e+f)  -> C22 += M6
    # M7 = (b-d)(g+h)  -> C11 += M7
    
    # A indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # B indices: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1.0  # a
    U[3, 0] = 1.0  # d
    V[0, 0] = 1.0  # e
    V[3, 0] = 1.0  # h
    W[0, 0] = 1.0  # C11
    W[3, 0] = 1.0  # C22
    
    # M2 = (c+d)e
    U[2, 1] = 1.0  # c
    U[3, 1] = 1.0  # d
    V[0, 1] = 1.0  # e
    W[1, 1] = 1.0  # C21
    W[3, 1] = -1.0 # C22
    
    # M3 = a(f-h)
    U[0, 2] = 1.0  # a
    V[1, 2] = 1.0  # f
    V[3, 2] = -1.0 # h
    W[2, 2] = 1.0  # C12
    W[3, 2] = 1.0  # C22
    
    # M4 = d(g-e)
    U[3, 3] = 1.0  # d
    V[2, 3] = 1.0  # g
    V[0, 3] = -1.0 # e
    W[0, 3] = 1.0  # C11
    W[1, 3] = 1.0  # C21
    
    # M5 = (a+b)h
    U[0, 4] = 1.0  # a
    U[1, 4] = 1.0  # b
    V[3, 4] = 1.0  # h
    W[0, 4] = -1.0 # C11
    W[2, 4] = 1.0  # C12
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1.0  # c
    U[0, 5] = -1.0 # a
    V[0, 5] = 1.0  # e
    V[1, 5] = 1.0  # f
    W[3, 5] = 1.0  # C22
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1.0  # b
    U[3, 6] = -1.0 # d
    V[2, 6] = 1.0  # g
    V[3, 6] = 1.0  # h
    W[0, 6] = 1.0  # C11
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build the rank-49 decomposition for 4x4 x 4x4 multiplication.
    Uses Strassen at the 4x4 level (7 block-products of 2x2 blocks),
    with each 2x2 x 2x2 block product done via Strassen (rank 7).
    Total rank = 7 * 7 = 49.
    
    Returns U (16, 49), V (16, 49), W (16, 49) for the 4x4 x 4x4 problem.
    U indexed by (i, j) in 4x4, V by (j, k) in 4x4, W by (k, i) in 4x4.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4x4 split into 2x2 blocks:
    # A block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bj, bk): rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    # C block (bk, bi): rows 2*bk to 2*bk+1, cols 2*bi to 2*bi+1
    
    # Strassen at 4x4 level (7 block-products):
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11       -> C21 += M2, C22 -= M2
    # M3 = A11(B12-B22)       -> C12 += M3, C22 += M3
    # M4 = A22(B21-B11)       -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22       -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # For each big product r (0-6), and each small Strassen product s (0-6):
    # We need to determine which A-blocks, B-blocks, C-blocks are involved.
    
    # Define the 7 big products as:
    # (list of (A_block_row, A_block_col, coeff_a),
    #  list of (B_block_row, B_block_col, coeff_b),
    #  list of (C_block_row, C_block_col, coeff_c))
    
    big_products = [
        # M1 = (A11+A22)(B11+B22)
        ([((0,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0), ((1,1), 1.0)]),
        # M2 = (A21+A22)B11
        ([((1,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0)],
         [((0,1), 1.0), ((1,1), -1.0)]),
        # M3 = A11(B12-B22)
        ([((0,0), 1.0)],
         [((0,1), 1.0), ((1,1), -1.0)],
         [((1,0), 1.0), ((1,1), 1.0)]),
        # M4 = A22(B21-B11)
        ([((1,1), 1.0)],
         [((1,0), 1.0), ((0,0), -1.0)],
         [((0,0), 1.0), ((0,1), 1.0)]),
        # M5 = (A11+A12)B22
        ([((0,0), 1.0), ((0,1), 1.0)],
         [((1,1), 1.0)],
         [((0,0), -1.0), ((1,0), 1.0)]),
        # M6 = (A21-A11)(B11+B12)
        ([((1,0), 1.0), ((0,0), -1.0)],
         [((0,0), 1.0), ((0,1), 1.0)],
         [((1,1), 1.0)]),
        # M7 = (A12-A22)(B21+B22)
        ([((0,1), 1.0), ((1,1), -1.0)],
         [((1,0), 1.0), ((1,1), 1.0)],
         [((0,0), 1.0)]),
    ]
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    for r, (a_blocks, b_blocks, c_blocks) in enumerate(big_products):
        for s in range(7):
            col = r * 7 + s
            
            # For each combination of A-block, B-block, C-block:
            for (ar, ac), ca in a_blocks:
                for (br, bc), cb in b_blocks:
                    for (cr, cc), cc_c in c_blocks:
                        # The small Strassen product s operates on the 2x2 blocks.
                        # U2[i2*2+j2, s] gives the coefficient for A element (i2, j2) in block (ar, ac)
                        # V2[j2*2+k2, s] gives the coefficient for B element (j2, k2) in block (br, bc)
                        # W2[k2*2+i2, s] gives the coefficient for C element (k2, i2) in block (cr, cc)
                        
                        for i2 in range(2):
                            for j2 in range(2):
                                u_coeff_a = U2[i2 * 2 + j2, s]
                                if abs(u_coeff_a) > 1e-15:
                                    i_global = 2 * ar + i2
                                    j_global = 2 * ac + j2
                                    U[i_global * 4 + j_global, col] += ca * u_coeff_a
                        
                        for j2 in range(2):
                            for k2 in range(2):
                                v_coeff_b = V2[j2 * 2 + k2, s]
                                if abs(v_coeff_b) > 1e-15:
                                    j_global = 2 * br + j2
                                    k_global = 2 * bc + k2
                                    V[j_global * 4 + k_global, col] += cb * v_coeff_b
                        
                        for k2 in range(2):
                            for i2 in range(2):
                                w_coeff_c = W2[k2 * 2 + i2, s]
                                if abs(w_coeff_c) > 1e-15:
                                    k_global = 2 * cr + k2
                                    i_global = 2 * cc + i2
                                    W[k_global * 4 + i_global, col] += cc_c * w_coeff_c
    
    return U, V, W


def _build_4x4x4x9_decomposition():
    """
    Build the full decomposition for 4x4 x 4x9.
    Split into:
      - 4x4 x 4x4 for cols 0-3 (Strassen, rank 49)
      - 4x4 x 4x4 for cols 4-7 (Strassen, rank 49)
      - 4x4 x 4x1 for col 8 (schoolbook, rank 16)
    Total rank = 49 + 49 + 16 = 114.
    """
    U1, V1, W1 = _build_4x4x4_strassen()  # cols 0-3
    U2, V2, W2 = _build_4x4x4_strassen()  # cols 4-7
    
    # For the 4x4 x 4x1 part (col 8), use schoolbook
    n, m, p1 = 4, 4, 1
    U3 = np.zeros((n * m, n * m * p1), dtype=np.float64)
    V3 = np.zeros((m * p1, n * m * p1), dtype=np.float64)
    W3 = np.zeros((p1 * n, n * m * p1), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            k = 0
            U3[i * m + j, col] = 1.0
            V3[j * p1 + k, col] = 1.0
            W3[k * n + i, col] = 1.0
            col += 1
    
    # Now combine:
    # U has shape (16, 114)
    # V has shape (36, 114)  [m*p = 4*9 = 36]
    # W has shape (36, 114)  [p*n = 9*4 = 36]
    
    U = np.zeros((16, 114), dtype=np.float64)
    V = np.zeros((36, 114), dtype=np.float64)
    W = np.zeros((36, 114), dtype=np.float64)
    
    # Part 1: cols 0-3
    U[:, 0:49] = U1
    # V1 has shape (16, 49) for 4x4 x 4x4, but we need V for 4x9.
    # V1[j*4+k, r] -> V[j*9+k, r] for k in 0-3
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V1[j * 4 + k, :]
    # W1 has shape (16, 49) for 4x4 x 4x4, W1[k*4+i, r] -> W[k*4+i, r] for k in 0-3
    # W index for 4x9 is k*4+i where k is 0-8
    W[0:16, 0:49] = W1
    
    # Part 2: cols 4-7
    U[:, 49:98] = U2
    for j in range(4):
        for k in range(4):
            V[j * 9 + (k + 4), 49:98] = V2[j * 4 + k, :]
    W[16:32, 49:98] = W2
    
    # Part 3: col 8
    U[:, 98:114] = U3
    for j in range(4):
        V[j * 9 + 8, 98:114] = V3[j * 1 + 0, :]
    W[32:36, 98:114] = W3
    
    return U, V, W


def _validate(U, V, W, n, m, p, tol=1e-9):
    """
    Check whether the factor matrices exactly decompose the matmul tensor:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p), dtype=np.float64)
    if not np.allclose(T, expected, atol=tol, rtol=tol):
        max_err = np.max(np.abs(T - expected))
        return False
    return True


def _schoolbook_full():
    """Return the trivial schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140.0
    
    # Build the Strassen-based decomposition (rank 114)
    U, V, W = _build_4x4x4x9_decomposition()
    
    # Validate the result
    if not _validate(U, V, W, n, m, p):
        # Fall back to schoolbook if validation fails
        U, V, W = _schoolbook_full()
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
