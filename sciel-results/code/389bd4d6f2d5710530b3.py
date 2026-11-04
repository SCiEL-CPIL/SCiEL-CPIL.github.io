import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for M_{n,m,p} with trace convention W[k*n+i, r]."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_222():
    """
    Strassen decomposition for M_{2,2,2} with trace convention W[k*2+i, r].
    Returns U (4x7), V (4x7), W (4x7).
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen formulas for C = A*B where A,B are 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    #
    # Indexing: A[i*m+j], B[j*p+k], C[k*n+i] with n=m=p=2
    # U indices: i*m+j -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p+k -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: k*n+i -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00 (idx 0) +1, a11 (idx 3) +1
    # V: b00 (idx 0) +1, b11 (idx 3) +1
    # W: c00 (idx 0) +1, c11 (idx 3) +1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    # U: a10 (idx 2) +1, a11 (idx 3) +1
    # V: b00 (idx 0) +1
    # W: c10 (idx 2) +1, c11 (idx 3) -1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00(b10-b11)
    # U: a00 (idx 0) +1
    # V: b10 (idx 2) +1, b11 (idx 3) -1
    # W: c01 (idx 1) +1, c11 (idx 3) +1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11(b01-b10)
    # U: a11 (idx 3) +1
    # V: b01 (idx 1) +1, b10 (idx 2) -1
    # W: c00 (idx 0) +1, c10 (idx 2) +1
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a00+a01)b11
    # U: a00 (idx 0) +1, a01 (idx 1) +1
    # V: b11 (idx 3) +1
    # W: c00 (idx 0) -1, c01 (idx 1) +1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    # U: a10 (idx 2) +1, a00 (idx 0) -1
    # V: b00 (idx 0) +1, b01 (idx 1) +1
    # W: c11 (idx 3) +1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    # U: a01 (idx 1) +1, a11 (idx 3) -1
    # V: b10 (idx 2) +1, b11 (idx 3) +1
    # W: c00 (idx 0) +1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_444():
    """
    Build Strassen decomposition for M_{4,4,4} using 49 terms.
    Uses block Strassen: split into 2x2 blocks, apply Strassen to each block pair.
    """
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Get base Strassen 2x2x2
    SU, SV, SW = _build_strassen_222()
    
    # We have 2x2 blocks of A (4x4), 2x2 blocks of B (4x4), producing 2x2 blocks of C (4x4)
    # A blocks: A[0,0] (rows 0-1, cols 0-1), A[0,1] (rows 0-1, cols 2-3), A[1,0] (rows 2-3, cols 0-1), A[1,1] (rows 2-3, cols 2-3)
    # B blocks: B[0,0], B[0,1], B[1,0], B[1,1] similarly
    # C blocks: C[0,0], C[0,1], C[1,0], C[1,1] similarly
    #
    # Block Strassen:
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    #
    # We apply Strassen at the block level. Each "element" is a 2x2 matrix.
    # The 7 Strassen terms M1..M7 are each 2x2 matrix products computed via 2x2x2 Strassen.
    
    # Block indices: (block_row, block_col) for 2x2 block grid
    # For each of the 7 Strassen terms, we need to compute a 2x2x2 product of the block-level operands.
    
    # The block-level Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level term, we compute a 2x2x2 product using Strassen (7 terms each).
    # Total: 7 * 7 = 49 terms.
    
    # Helper to get block offset
    def a_offset(br, bc):
        """Offset in A (4x4) for block (br, bc) where br,bc in {0,1}"""
        return br * 2 * m + bc * 2  # row start * m + col start
    
    def b_offset(br, bc):
        """Offset in B (4x4) for block (br, bc)"""
        return br * 2 * p + bc * 2
    
    def c_offset(br, bc):
        """Offset in C (4x4) for block (br, bc), using W convention k*n+i"""
        # C block (br, bc) corresponds to rows (output) br*2..br*2+1, cols (input k) bc*2..bc*2+1
        # W index: k*n + i, where i is output row, k is input col
        # So for block (br, bc): i in br*2..br*2+1, k in bc*2..bc*2+1
        # The W entries for this block are at positions k*4+i for k in {bc*2, bc*2+1}, i in {br*2, br*2+1}
        return None  # We'll handle W differently
    
    # Block-level Strassen: for each of the 7 terms, define the A-block combination and B-block combination
    # term -> (list of (a_block_row, a_block_col, coeff), list of (b_block_row, b_block_col, coeff), c_block_output)
    # Actually, let's think of it as: for each Strassen term t, we have:
    #   A_combined = sum of A blocks with coefficients
    #   B_combined = sum of B blocks with coefficients
    #   Then M_t = A_combined * B_combined (2x2x2 product)
    #   And C blocks get contributions from M_t.
    
    # Let's define for each of the 7 terms:
    # term_t: A_parts = [(br_a, bc_a, coeff_a)], B_parts = [(br_b, bc_b, coeff_b)]
    # Then M_t is the 2x2x2 product of these combined blocks.
    
    terms = [
        # M1: (A00+A11)*(B00+B11)
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)]},
        # M2: (A10+A11)*B00
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)]},
        # M3: A00*(B10-B11)
        {'A': [(0,0,1)], 'B': [(1,0,1), (1,1,-1)]},
        # M4: A11*(B01-B10)
        {'A': [(1,1,1)], 'B': [(0,1,1), (1,0,-1)]},
        # M5: (A00+A01)*B11
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)]},
        # M6: (A10-A00)*(B00+B01)
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)]},
        # M7: (A01-A11)*(B10+B11)
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)]},
    ]
    
    # C block contributions from each M_t:
    # C00 += M1, C00 += M4, C00 -= M5, C00 += M7
    # C01 += M3, C01 += M5
    # C10 += M2, C10 += M4
    # C11 += M1, C11 -= M2, C11 += M3, C11 += M6
    c_contrib = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],  # M1+M4-M5+M7
        (0,1): [(2,1), (4,1)],                   # M3+M5
        (1,0): [(1,1), (3,1)],                   # M2+M4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],    # M1-M2+M3+M6
    }
    
    col = 0
    for t_idx in range(7):
        term = terms[t_idx]
        # For this block-level term, we compute a 2x2x2 Strassen product
        # The 2x2x2 Strassen has 7 sub-terms
        for s_idx in range(7):
            # SU[s_idx] gives coefficients for A entries (2x2), SV for B entries, SW for C entries
            # A entry (i,j) in the 2x2 sub-block corresponds to A block (br_a, bc_a) entry (i,j)
            # which maps to A global index: (br_a*2+i)*m + (bc_a*2+j)
            
            # For each nonzero in SU[s_idx]:
            for a_local_i in range(2):
                for a_local_j in range(2):
                    coeff_a = SU[a_local_i * 2 + a_local_j][s_idx]
                    if coeff_a == 0:
                        continue
                    for (br_a, bc_a, coeff_block_a) in term['A']:
                        global_a_row = br_a * 2 + a_local_i
                        global_a_col = bc_a * 2 + a_local_j
                        a_idx = global_a_row * m + global_a_col
                        U[a_idx][col] += coeff_a * coeff_block_a
            
            for b_local_j in range(2):
                for b_local_k in range(2):
                    coeff_b = SV[b_local_j * 2 + b_local_k][s_idx]
                    if coeff_b == 0:
                        continue
                    for (br_b, bc_b, coeff_block_b) in term['B']:
                        global_b_row = br_b * 2 + b_local_j
                        global_b_col = bc_b * 2 + b_local_k
                        b_idx = global_b_row * p + global_b_col
                        V[b_idx][col] += coeff_b * coeff_block_b
            
            # W: for each C block that gets contribution from this M_t
            for (c_br, c_bc) in [(0,0), (0,1), (1,0), (1,1)]:
                for (t_contrib, coeff_c) in c_contrib[(c_br, c_bc)]:
                    if t_contrib != t_idx:
                        continue
                    for c_local_k in range(2):
                        for c_local_i in range(2):
                            coeff_w = SW[c_local_k * 2 + c_local_i][s_idx]
                            if coeff_w == 0:
                                continue
                            global_c_k = c_bc * 2 + c_local_k
                            global_c_i = c_br * 2 + c_local_i
                            w_idx = global_c_k * n + global_c_i
                            W[w_idx][col] += coeff_w * coeff_c
            
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Test on a random A, B
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_actual = np.zeros((n, p))
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        # u_col is n*m vector, reshape to (n,m)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # Contribution: sum over j of u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # = trace-like: for each i,k, sum_j u_mat[i,j]*v_mat[j,k] * w_mat[k,i]
        # This is (u_mat @ v_mat) * w_mat element-wise, then sum over i,k? No.
        # Actually: sum_j u[i,j]*v[j,k] = (u@v)[i,k], then multiply by w[k,i] and sum over i,k?
        # No, the tensor entry T[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
        # C[i,k] = sum_j T[i,j,k] = sum_j U[i,j,r]*V[j,k,r]*W[k,i,r]
        # = W[k,i,r] * sum_j U[i,j,r]*V[j,k,r] = W[k,i,r] * (U_r @ V_r)[i,k]
        uv = u_mat @ v_mat  # (n, p)
        for i in range(n):
            for k in range(p):
                C_actual[i, k] += uv[i, k] * w_mat[k, i]
    
    if not np.allclose(C_actual, C_expected, atol=1e-10):
        return False, np.max(np.abs(C_actual - C_expected))
    return True, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    p1 = 4  # Number of columns to handle with Strassen
    p2 = p - p1  # Remaining columns for schoolbook
    
    if p1 <= 0:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Part 1: Strassen for M_{n,m,p1} = M_{4,4,4}
    U1, V1, W1 = _build_strassen_444()
    rank1 = len(U1[0])
    
    # Part 2: Schoolbook for remaining p2 columns
    # We need to extend the schoolbook to only cover columns p1..p-1
    U2 = [[0] * (n * m * p2) for _ in range(n * m)]
    V2 = [[0] * (n * m * p2) for _ in range(m * p)]
    W2 = [[0] * (n * m * p2) for _ in range(p * n)]
    col2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):
                U2[i * m + j][col2] = 1
                V2[j * p + k][col2] = 1
                W2[k * n + i][col2] = 1
                col2 += 1
    rank2 = n * m * p2
    
    # Combine: concatenate columns
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
        for r in range(rank2):
            U[i][rank1 + r] = U2[i][r]
    
    for i in range(m * p):
        for r in range(rank1):
            V[i][r] = V1[i][r]
        for r in range(rank2):
            V[i][rank1 + r] = V2[i][r]
    
    for i in range(p * n):
        for r in range(rank1):
            W[i][r] = W1[i][r]
        for r in range(rank2):
            W[i][rank1 + r] = W2[i][r]
    
    # Verify
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
