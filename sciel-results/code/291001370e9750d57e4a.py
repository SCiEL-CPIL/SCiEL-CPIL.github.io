"""Hybrid Strassen+schoolbook decomposition for 4x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen multiplication (rank 7).
    
    Indexing convention:
    - U has shape (n*m, rank) = (4, 7), indexed by [i*m + j, r]
    - V has shape (m*p, rank) = (4, 7), indexed by [j*p + k, r]
    - W has shape (p*n, rank) = (4, 7), indexed by [k*n + i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10 + a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00 + a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a10 - a00)(b00 + b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a11 - a10)(b10 + b11)
    U[3, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_4x4x6_hybrid():
    """Build hybrid Strassen+schoolbook decomposition for 4x4 * 4x6.
    
    Strategy:
    - Split B (4x6) into B_left (4x4) and B_right (4x2)
    - C_left (4x4) = A * B_left: use block-level Strassen (7 terms), each term is a 2x2x2x3 problem
    - For each Strassen term: inner Strassen 2x2x2 (rank 7) for first 2 cols + schoolbook 2x2x1 (rank 4) for last col = rank 11
    - C_right (4x2) = A * B_right: schoolbook, rank 32
    - Total rank: 7 * 11 + 32 = 109? No wait...
    
    Actually let me reconsider. The 4x4*4x4 part with block Strassen:
    - 7 block-level terms, each computing a 2x2 * 2x2 = 2x2 matrix product
    - Each 2x2*2x2 product uses inner Strassen: rank 7
    - Total for 4x4*4x4: 7 * 7 = 49
    
    For the 4x4*4x2 part (right 2 columns):
    - Schoolbook: 4*4*2 = 32
    
    Total: 49 + 32 = 81
    
    But the island 2 finding says rank 77. Let me re-read:
    "Replacing schoolbook 2x2x3 (rank 12) per Strassen term with inner Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = rank 11 per term, reducing total from 84 to 77"
    
    So the approach is:
    - Split B (4x6) into 3 blocks of 2 columns each: B0 (4x2), B1 (4x2), B2 (4x2)
    - Use block-level Strassen on the 4x4 * 4x6 where B is viewed as 2x3 blocks of 2x2 matrices
    - Wait, that doesn't quite work because Strassen needs square blocks.
    
    Let me re-think. The 4x4 * 4x6 can be viewed as:
    - A is 4x4, split into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    - B is 4x6, split into 2x3 blocks of 2x2: B = [[B00, B01, B02], [B10, B11, B12]]
    - C is 4x6, split into 2x3 blocks of 2x2: C = [[C00, C01, C02], [C10, C11, C12]]
    
    For the first two column blocks of B (4x4 part), use Strassen at block level:
    - 7 terms, each term multiplies a 2x2 linear combo of A blocks by a 2x2 linear combo of B blocks
    - Each such multiplication is a 2x2 * 2x2 = 2x2 product, done with inner Strassen (rank 7)
    - Total: 7 * 7 = 49 for the first 4 columns
    
    For the last column block of B (4x2 part), use schoolbook:
    - 4 * 4 * 2 = 32
    
    Total: 49 + 32 = 81
    
    But the finding says 77. Let me re-read more carefully:
    "reducing total from 84 to 77"
    
    84 = 7 * 12 (schoolbook 2x2x3 per Strassen term)
    77 = 7 * 11 (Strassen 2x2x2 + schoolbook 2x2x1 per Strassen term)
    
    So the approach is:
    - View B as having 3 column-blocks, each 2 columns wide
    - For each Strassen block-level term, instead of doing a full 2x2 * 2x3 = 2x3 schoolbook (rank 12),
      do a 2x2 * 2x2 Strassen (rank 7) for the first 2 output columns + 2x2 * 2x1 schoolbook (rank 4) for the last output column
    - This gives rank 11 per Strassen term instead of rank 12
    
    Wait, but that would mean the Strassen is applied to a 2x2 * 2x3 problem where the output is 2x3.
    The "2x2x3" refers to multiplying a 2x2 matrix by a 2x3 matrix to get a 2x3 result.
    Schoolbook rank for 2x2x3 is 2*2*3 = 12.
    
    The hybrid: split the 2x3 output into first 2 columns (2x2) and last 1 column (2x1).
    - First 2 columns: 2x2 * 2x2 product using Strassen = rank 7
    - Last 1 column: 2x2 * 2x1 product using schoolbook = rank 4
    - Total per term: 7 + 4 = 11
    
    But wait, the input B block is 2x3 (2 rows, 3 cols of 2x2 sub-blocks). The first 2 columns of B form a 2x2 block matrix, and the last 1 column is a 2x1 block matrix.
    
    Actually, I think the structure is:
    - A is 4x4 = 2x2 blocks of 2x2
    - B is 4x6 = 2x3 blocks of 2x2 (2 row-blocks, 3 col-blocks)
    - C is 4x6 = 2x3 blocks of 2x2
    
    At the block level, we want to compute C = A * B where A is 2x2 blocks, B is 2x3 blocks.
    
    For the first 2 column-blocks of B (which form a 2x2 block matrix), use Strassen:
    - 7 terms, each computing a 2x2 * 2x2 block product
    - Each block product is done with inner Strassen (rank 7)
    - Total: 7 * 7 = 49
    
    For the last 1 column-block of B (a 2x1 block matrix), use schoolbook:
    - 2 * 2 * 1 = 4 block-level multiplications, each 2x2 * 2x2 = rank 7 with inner Strassen? No...
    - Actually, for a 2x2 * 2x1 block multiplication (result is 2x1 block matrix), schoolbook is 2*2*1 = 4 terms, each term is a 2x2 * 2x2 = 2x2 product.
    - If each 2x2*2x2 uses inner Strassen: 4 * 7 = 28
    - Total: 49 + 28 = 77!
    
    Yes! That's it. So:
    - First 4 columns (2 col-blocks): block Strassen (7 terms) x inner Strassen (7) = 49
    - Last 2 columns (1 col-block): schoolbook at block level (4 terms) x inner Strassen (7) = 28
    - Total: 49 + 28 = 77
    
    Let me implement this.
    """
    n, m, p = 4, 4, 6
    U2, V2, W2 = _strassen_2x2()
    
    # Total rank: 7 * 7 + 4 * 7 = 49 + 28 = 77
    rank = 77
    U = np.zeros((n * m, rank))  # (16, 77)
    V = np.zeros((m * p, rank))  # (24, 77)
    W = np.zeros((p * n, rank))  # (24, 77)
    
    # Part 1: First 4 columns of B (col-blocks 0 and 1)
    # Block-level Strassen: 7 terms
    # Each term: inner Strassen 2x2x2: 7 scalar terms
    # Total: 49 terms
    
    for r in range(7):  # Block-level Strassen term
        for s in range(7):  # Inner Strassen term
            col = r * 7 + s
            
            # For block-level term r:
            # A_lin[a, b] = sum_{i, j} U2[i*2+j, r] * A_block[i, j][a, b]
            #             = sum_{i, j} U2[i*2+j, r] * A[2*i+a, 2*j+b]
            # B_lin[c, d] = sum_{j, k} V2[j*2+k, r] * B_block[j, k][c, d]
            #             = sum_{j, k} V2[j*2+k, r] * B[2*j+c, 2*k+d]
            # (for the first 2 col-blocks of B, so k in {0,1})
            
            # Inner Strassen term s computes:
            # M_s = (sum_{a, b} U2[a*2+b, s] * A_lin[a, b]) * (sum_{c, d} V2[c*2+d, s] * B_lin[c, d])
            # Contribution to C_block[f, e] = W2[e*2+f, s] * M_s
            # where f, e in {0,1} for the output block position
            
            # Expand A_lin[a, b]:
            # A_lin[a, b] = sum_{i, j} U2[i*2+j, r] * A[2*i+a, 2*j+b]
            # Coefficient of A[2*i+a, 2*j+b] in the first factor of M_s:
            #   sum_{a', b'} U2[a'*2+b', s] * (coefficient of A[2*i+a, 2*j+b] in A_lin[a', b'])
            #   = sum_{a', b'} U2[a'*2+b', s] * U2[i*2+j, r] * delta_{a, a'} * delta_{b, b'}
            #   = U2[a*2+b, s] * U2[i*2+j, r]
            
            # Similarly for B_lin[c, d]:
            # Coefficient of B[2*j2+c, 2*k+d] in the second factor of M_s:
            #   = V2[c*2+d, s] * V2[j2*2+k, r]
            
            # Contribution to C[2*f+a, 2*e+b] from term (r, s):
            # W2[e*2+f, s] * U2[a*2+b, s] * U2[i*2+j, r] * V2[c*2+d, s] * V2[j2*2+k, r]
            # for all a, b, i, j, c, d, j2, k, f, e
            
            for a in range(2):
                for b in range(2):
                    for i in range(2):
                        for j in range(2):
                            u_coeff = U2[a*2+b, s] * U2[i*2+j, r]
                            if u_coeff != 0:
                                a_idx = 2*i + a
                                j_idx = 2*j + b
                                for c in range(2):
                                    for d in range(2):
                                        for j2 in range(2):
                                            for k in range(2):
                                                v_coeff = V2[c*2+d, s] * V2[j2*2+k, r]
                                                if v_coeff != 0:
                                                    # j2 is the row-block of B, k is the col-block (0 or 1 for first 4 cols)
                                                    b_idx = 2*j2 + c
                                                    k_idx = 2*k + d
                                                    for f in range(2):
                                                        for e in range(2):
                                                            w_coeff = W2[e*2+f, s]
                                                            if w_coeff != 0:
                                                                # C[2*f+a, 2*e+b] but wait, the output block is C_block[f, e]
                                                                # C_block[f, e][a, b] = C[2*f+a, 2*e+b]
                                                                # But e here is the col-block index in the 2x2 output
                                                                # For the first 4 columns, col-block e corresponds to global col 2*e+b
                                                                c_row = 2*f + a
                                                                c_col = 2*e + b
                                                                w_idx = c_col * n + c_row  # W[k*n+i, r] where k=c_col, i=c_row
                                                                
                                                                U[a_idx * m + j_idx, col] += u_coeff
                                                                V[b_idx * p + k_idx, col] += v_coeff
                                                                W[w_idx, col] += w_coeff
    
    # Part 2: Last 2 columns of B (col-block 2)
    # Schoolbook at block level: 4 terms (2 row-blocks of A x 1 col-block of B)
    # Each term: inner Strassen 2x2x2: 7 scalar terms
    # Total: 28 terms
    
    offset = 49
    for i_block in range(2):  # Row-block of A
        for j_block in range(2):  # Col-block of A (which is row-block of B)
            # Term: A_block[i_block, j_block] * B_block[j_block, 2]
            # This is a 2x2 * 2x2 = 2x2 product, done with inner Strassen
            for s in range(7):
                col = offset + (i_block * 2 + j_block) * 7 + s
                
                # A_block[i_block, j_block][a, b] = A[2*i_block+a, 2*j_block+b]
                # B_block[j_block, 2][c, d] = B[2*j_block+c, 2*2+d] = B[2*j_block+c, 4+d]
                # C_block[i_block, 2][a, d] = C[2*i_block+a, 4+d]
                
                for a in range(2):
                    for b in range(2):
                        u_coeff = U2[a*2+b, s]
                        if u_coeff != 0:
                            a_idx = 2*i_block + a
                            j_idx = 2*j_block + b
                            for c in range(2):
                                for d in range(2):
                                    v_coeff = V2[c*2+d, s]
                                    if v_coeff != 0:
                                        b_idx = 2*j_block + c
                                        k_idx = 4 + d  # col-block 2, so global col 4+d
                                        for f in range(2):
                                            for e in range(2):
                                                w_coeff = W2[e*2+f, s]
                                                if w_coeff != 0:
                                                    # C_block[i_block, 2][a, d] = C[2*i_block+a, 4+d]
                                                    # W indexing: W[k*n+i, r] where k=col, i=row
                                                    c_row = 2*i_block + a
                                                    c_col = 4 + d
                                                    w_idx = c_col * n + c_row
                                                    
                                                    U[a_idx * m + j_idx, col] += u_coeff
                                                    V[b_idx * p + k_idx, col] += v_coeff
                                                    W[w_idx, col] += w_coeff
    
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    U, V, W = _build_4x4x6_hybrid()
    
    # Verify correctness with a random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        # u_vec has 16 entries: A[i*m+j]
        # v_vec has 24 entries: B[j*p+k]
        # w_vec has 24 entries: C[k*n+i]
        A_flat = A_test.flatten()  # A[i*m+j]
        B_flat = B_test.flatten()  # B[j*p+k]
        scalar = np.dot(u_vec, A_flat) * np.dot(v_vec, B_flat)
        C_flat = np.zeros(p * n)
        C_flat += scalar * w_vec
        C_computed = C_flat.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
