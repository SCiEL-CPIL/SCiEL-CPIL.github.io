"""Matrix multiplication 4x5x6 with Strassen-based block decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2(A, B):
    """
    Multiply two 2x2 matrices using Strassen's algorithm (7 multiplications).
    A and B are 2x2 lists (or arrays). Returns 2x2 result.
    All operations use Fraction for exactness.
    """
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    a11, a12 = A[0][0], A[0][1]
    a21, a22 = A[1][0], A[1][1]
    b11, b12 = B[0][0], B[0][1]
    b21, b22 = B[1][0], B[1][1]

    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    return [[c11, c12], [c21, c22]]


def _build_strassen_2x2_factors():
    """
    Build the 7 rank-1 factors for 2x2 matrix multiplication via Strassen.
    
    The tensor T[i,j,k] = delta_{i,k} * (A_{i,j} * B_{j,k})
    We want: T = sum_{c=1}^{7} u_c ⊗ v_c ⊗ w_c
    where (u_c)_{i,j} * (v_c)_{j,k} * (w_c)_{i,k} sums to T[i,j,k].
    
    Using the standard Strassen decomposition:
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m2 + m3 + m6
    
    where:
    m1 = (A00+A11)(B00+B11)
    m2 = (A10+A11)B00
    m3 = A00(B01-B11)
    m4 = A11(B10-B00)
    m5 = (A00+A01)B11
    m6 = (A10-A00)(B00+B01)
    m7 = (A01-A11)(B10+B11)
    
    For each m_c, we need u_c, v_c, w_c such that:
    sum_c u_c[i,j] * v_c[j,k] * w_c[i,k] = T[i,j,k] = delta(i,k)
    
    For a given m_c = (linear_comb_A) * (linear_comb_B):
    u_c[i,j] = coefficient of A[i,j] in linear_comb_A
    v_c[j,k] = coefficient of B[j,k] in linear_comb_B
    w_c[i,k] = coefficient of m_c in C[i,k]
    """
    # Each factor: (u, v, w) where u is 2x2 (A part), v is 2x2 (B part), w is 2x2 (C part)
    # u[i,j] = coeff of A[i,j], v[j,k] = coeff of B[j,k], w[i,k] = coeff in C[i,k]
    
    factors = []
    
    # m1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 0], [0, 1]]
    w1 = [[1, 0], [0, 1]]
    factors.append((u1, v1, w1))
    
    # m2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    u2 = [[0, 0], [1, 1]]
    v2 = [[1, 0], [0, 0]]
    w2 = [[0, 0], [1, -1]]
    factors.append((u2, v2, w2))
    
    # m3 = A00(B01-B11) -> C01 += 1, C11 += 1
    u3 = [[1, 0], [0, 0]]
    v3 = [[0, 1], [-1, 0]]
    w3 = [[0, 1], [0, 1]]
    factors.append((u3, v3, w3))
    
    # m4 = A11(B10-B00) -> C00 += 1, C10 += 1
    u4 = [[0, 0], [0, 1]]
    v4 = [[-1, 0], [1, 0]]
    w4 = [[1, 0], [1, 0]]
    factors.append((u4, v4, w4))
    
    # m5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    u5 = [[1, 1], [0, 0]]
    v5 = [[0, 0], [0, 1]]
    w5 = [[-1, 1], [0, 0]]
    factors.append((u5, v5, w5))
    
    # m6 = (A10-A00)(B00+B01) -> C11 += 1
    u6 = [[-1, 0], [1, 0]]
    v6 = [[1, 1], [0, 0]]
    w6 = [[0, 0], [0, 1]]
    factors.append((u6, v6, w6))
    
    # m7 = (A01-A11)(B10+B11) -> C00 += 1
    u7 = [[0, 1], [0, -1]]
    v7 = [[0, 0], [1, 1]]
    w7 = [[1, 0], [0, 0]]
    factors.append((u7, v7, w7))
    
    return factors


def _verify_strassen_factors(factors):
    """Verify that the Strassen factors correctly decompose 2x2x2 tensor."""
    # Build tensor T[i,j,k] = 1 if i==k else 0
    n = 2
    T = np.zeros((n, n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[i, j, k] = 1 if i == k else 0
    
    # Reconstruct from factors
    T_rec = np.zeros((n, n, n), dtype=int)
    for u, v, w in factors:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    T_rec[i, j, k] += u[i][j] * v[j][k] * w[i][k]
    
    return np.array_equal(T, T_rec)


def _build_block_decomposition(n, m, p, block_size=2):
    """
    Build a decomposition for n x p = A(n x m) * B(m x p) using block Strassen.
    
    Strategy:
    - Partition A into blocks of size block_size x block_size (padded if needed)
    - Partition B similarly
    - Use Strassen for 2x2 blocks where possible
    - Use schoolbook for remaining 1x1 blocks
    
    For 4x5*5x6:
    - n=4, m=5, p=6
    - A: 4x5, B: 5x6
    - We can partition:
      - Rows of A: 4 = 2+2 (two 2x? blocks)
      - Cols of A / Rows of B: 5 = 2+2+1
      - Cols of B: 6 = 2+2+2
    
    Let's use a more systematic approach:
    - Split A into row-blocks and col-blocks
    - Split B into row-blocks and col-blocks
    - For each pair of (A-block, B-block), use Strassen if 2x2, else schoolbook
    
    Actually, let's think of it as:
    A is n x m, B is m x p, C is n x p.
    
    Partition:
    - A rows: [0:2, 2:4] (2 blocks of size 2)
    - A cols / B rows: [0:2, 2:4, 4:5] (blocks of size 2, 2, 1)
    - B cols: [0:2, 2:4, 4:6] (3 blocks of size 2)
    
    For each (row_block_a, col_block_a, col_block_b):
      - A_sub: 2 x block_size_col
      - B_sub: block_size_col x 2
      - C_sub: 2 x 2
      
      If block_size_col == 2: use Strassen (7 factors)
      If block_size_col == 1: schoolbook (2 factors)
    
    Total factors:
    - 2 (row blocks) * 3 (col blocks) * [7 if size 2, 2 if size 1]
    = 2 * (7 + 7 + 2) = 2 * 16 = 32
    
    Wait, that's not right. Let me reconsider.
    
    Actually, the decomposition is over the full tensor. Let me think more carefully.
    
    The matrix multiplication tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    is defined as T[i,j,k] = 1 if the (i,k) entry of C = A*B depends on A[i,j]*B[j,k].
    
    Actually T[i,j,k] = delta(i,k) is wrong for rectangular. 
    T[i,j,k] = 1 always (it's the standard matrix mult tensor).
    
    The decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-1 term: u[i,j] * v[j,k] * w[i,k] contributes u[i,j]*v[j,k]*w[i,k] to C[i,k].
    
    For schoolbook: for each j, u[i,j]=delta(i,?)*... 
    
    Let me restart with a cleaner approach.
    
    Schoolbook for n x m * m x p:
    For each j in range(m):
        u[i,j] = 1 for all i, 0 otherwise  -> actually u is indexed by (i,j)
        v[j,k] = 1 for all k
        w[i,k] = 1 for all i,k
        
    No wait. u has shape (n*m, rank), v has shape (m*p, rank), w has shape (p*n, rank).
    
    u[row_a, c] where row_a = i*m + j
    v[row_b, c] where row_b = j*p + k  
    w[row_c, c] where row_c = k*n + i
    
    For schoolbook, factor c corresponds to (i,j,k):
    u[i*m+j, c] = 1
    v[j*p+k, c] = 1
    w[k*n+i, c] = 1
    
    For Strassen on a 2x2x2 sub-block (i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}):
    We have 7 factors. For factor f with (u_f, v_f, w_f) each 2x2:
    For local indices (ii, jj, kk) in {0,1}^3:
    u[(i0+ii)*m + (j0+jj), c] = u_f[ii][jj]
    v[(j0+jj)*p + (k0+kk), c] = v_f[jj][kk]
    w[(k0+kk)*n + (i0+ii), c] = w_f[ii][kk]
    
    Let me implement this.
    """
    # Partition boundaries
    # A rows: 0..n
    # A cols / B rows: 0..m  
    # B cols: 0..p
    
    # Choose block partition
    # For n=4, m=5, p=6:
    # A rows: [0, 2, 4] -> blocks [0:2], [2:4]
    # Middle: [0, 2, 4, 5] -> blocks [0:2], [2:4], [4:5]
    # B cols: [0, 2, 4, 6] -> blocks [0:2], [2:4], [4:6]
    
    # General: split into blocks of size block_size, last one may be smaller
    def make_blocks(total, bs):
        blocks = []
        start = 0
        while start < total:
            end = min(start + bs, total)
            blocks.append((start, end))
            start = end
        return blocks
    
    a_row_blocks = make_blocks(n, block_size)
    mid_blocks = make_blocks(m, block_size)
    b_col_blocks = make_blocks(p, block_size)
    
    # Get Strassen factors
    strassen_factors = _build_strassen_2x2_factors()
    
    # Build U, V, W as lists of lists
    # First, determine total rank
    rank = 0
    for (i0, i1) in a_row_blocks:
        for (j0, j1) in mid_blocks:
            for (k0, k1) in b_col_blocks:
                a_h = i1 - i0
                m_h = j1 - j0
                b_w = k1 - k0
                if a_h == 2 and m_h == 2 and b_w == 2:
                    rank += 7
                else:
                    rank += a_h * m_h * b_w
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, i1) in a_row_blocks:
        for (j0, j1) in mid_blocks:
            for (k0, k1) in b_col_blocks:
                a_h = i1 - i0
                m_h = j1 - j0
                b_w = k1 - k0
                
                if a_h == 2 and m_h == 2 and b_w == 2:
                    # Use Strassen
                    for u_f, v_f, w_f in strassen_factors:
                        for ii in range(2):
                            for jj in range(2):
                                U[(i0 + ii) * m + (j0 + jj)][col] = u_f[ii][jj]
                        for jj in range(2):
                            for kk in range(2):
                                V[(j0 + jj) * p + (k0 + kk)][col] = v_f[jj][kk]
                        for kk in range(2):
                            for ii in range(2):
                                W[(k0 + kk) * n + (i0 + ii)][col] = w_f[ii][kk]
                        col += 1
                else:
                    # Schoolbook
                    for ii in range(a_h):
                        for jj in range(m_h):
                            for kk in range(b_w):
                                U[(i0 + ii) * m + (j0 + jj)][col] = 1
                                V[(j0 + jj) * p + (k0 + kk)][col] = 1
                                W[(k0 + kk) * n + (i0 + ii)][col] = 1
                                col += 1
    
    return U, V, W, rank


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses block Strassen decomposition to reduce rank below schoolbook (120).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the block Strassen decomposition
    U, V, W, rank = _build_block_decomposition(n, m, p, block_size=2)
    
    # Verify correctness
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # Check: sum_c U[:,c] * V[:,c] * W[:,c] should give the tensor
    # Tensor T has T[i*m+j, j*p+k, k*n+i] structure
    # Actually, we need to verify that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Quick verification with a subset
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    verified = False
                    break
        if not verified:
            break
    
    if not verified:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        rank = n * m * p
    else:
        rank = len(U[0]) if U else 0
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but guard
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
