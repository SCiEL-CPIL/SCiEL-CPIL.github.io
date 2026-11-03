"""
Rank-28 exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
Uses Strassen's algorithm on 2x2 blocks, with runtime verification and fallback to schoolbook (rank 32).
"""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=4, p=4, num_trials=3, seed=42):
    """
    Verify that U @ diag(coeff) @ V^T gives correct C = A @ B for random inputs.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: C[k*n+i, j] = sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    i.e., C = (W^T @ diag(1) @ U^T) ... actually:
    C[i, j] = sum_k sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Let's think more carefully.
    A is n x m, B is m x p, C is n x p.
    C[i, j] = sum_k A[i, k] * B[k, j]
    
    U has shape (n*m, rank). U[i*m + k, c] corresponds to A[i, k].
    V has shape (m*p, rank). V[k*p + j, c] corresponds to B[k, j].
    W has shape (p*n, rank). W[j*n + i, c] corresponds to C[j, i]... 
    
    Wait, the task says: "The output factor follows the official trace convention W[k*n+i,r]."
    Hmm, let me re-read: "W[k*n+i,r]" - this suggests W is indexed by (k, i) where k is the column of C (0..p-1) and i is the row of C (0..n-1)? That would be W[k*n+i, r] for C[i, k]? No, that's p*n entries.
    
    Actually, looking at the seed program:
    W[k * n + i][column] = 1
    where k is 0..p-1 and i is 0..n-1. So W has p*n rows.
    The output C is n x p. C[i, k] = sum_j A[i,j] B[j,k].
    In the schoolbook, for each (i,j,k): U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    So C[i, k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    = sum_j A[i,j] B[j,k] * 1.
    
    So the convention is: C[i, k] = sum_{j, c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where k is the column index of C (0..p-1), i is the row index of C (0..n-1).
    W is indexed by [k*n + i, c] i.e. k is the outer index (column of C), i is inner (row of C).
    """
    import random
    random.seed(seed)
    rank = len(U[0])
    for _ in range(num_trials):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # True C
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Computed C
        C_comp = [[0]*p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    u_val = U[i*m + (0)][c]  # placeholder, need to sum over j
                    # Actually: C[i,k] += sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    for j in range(m):
                        C_comp[i][k] += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
        for i in range(n):
            for k in range(p):
                if C_true[i][k] != C_comp[i][k]:
                    return False
    return True


def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    U entries for A (2x2), V entries for B (2x2), and W entries for C (2x2).
    We'll represent as lists of 7 terms, each term being (u_coeffs, v_coeffs, w_coeffs)
    where each is a 2-tuple (or we flatten to 4 entries each).
    
    Standard Strassen:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B10 - B11)
    M4 = A11 (B01 - B00)
    M5 = (A00 + A01) B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For each of the 7 products, define:
    # U_coeffs: 4 values for A00, A01, A10, A11 (order: [i*m+j] with m=2, so A00=0, A01=1, A10=2, A11=3)
    # V_coeffs: 4 values for B00, B01, B10, B11 (order: [k*p+j] with p=2, so B00=0, B01=1, B10=2, B11=3)
    # W_coeffs: 4 values for C00, C01, C10, C11 (order: [k*n+i] with n=2, so C00=0, C01=1, C10=2, C11=3)
    # Wait, W is indexed by [k*n + i]. k is column of C, i is row of C.
    # C[i,k] where i in {0,1}, k in {0,1}.
    # k=0, i=0: C[0,0] -> index 0*2+0 = 0
    # k=0, i=1: C[1,0] -> index 0*2+1 = 1
    # k=1, i=0: C[0,1] -> index 1*2+0 = 2
    # k=1, i=1: C[1,1] -> index 1*2+1 = 3
    # So W order is: [C00, C10, C01, C11] in terms of (row, col) = (i, k).
    
    terms = []
    # M1: (A00+A11)(B00+B11)
    terms.append((
        [1, 0, 0, 1],  # A00, A01, A10, A11
        [1, 0, 0, 1],  # B00, B01, B10, B11
        [1, 0, 0, 1],  # C00, C10, C01, C11 -> M1 contributes to C00 (+1), C11 (+1)... wait
    ))
    # Hmm, I need to be more careful. Let me define W_coeffs as contribution to each C entry.
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1: C00 += 1, C11 += 1, others 0
    # W order: [C00, C10, C01, C11]
    # M1: W = [1, 0, 0, 1]
    
    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3: A00(B10-B11) -> C01 += 1, C11 += 1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        # M4: A11(B01-B00) -> C00 -= 1, C10 += 1
        ([0, 0, 0, 1], [0, 1, -1, 0], [-1, 1, 0, 0]),
        # M5: (A00+A01)B11 -> C00 += 1, C01 += 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    return terms


def _build_strassen_2x4x4():
    """
    Build the 28-rank decomposition for 2x4 x 4x4.
    Split A (2x4) into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
    Split B (4x4) into four 2x2 blocks: B_TL (rows 0,1; cols 0,1), B_TR (rows 0,1; cols 2,3),
                                            B_BL (rows 2,3; cols 0,1), B_BR (rows 2,3; cols 2,3)
    C (2x4) has two 2x2 blocks: C_left (cols 0,1), C_right (cols 2,3)
    
    C_left = A_left @ B_TL + A_right @ B_BL
    C_right = A_left @ B_TR + A_right @ B_BR
    
    For each 2x2 @ 2x2 product, we use 7 terms.
    Total: 4 products * 7 terms = 28 terms.
    """
    strassen_terms = _strassen_2x2()
    
    rank = 28
    n, m, p = 2, 4, 4
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    col = 0
    # Block assignments:
    # (A_block, B_block, C_block) where:
    # A_block: 0=left (cols 0,1), 1=right (cols 2,3)
    # B_block: 0=TL (rows 0,1; cols 0,1), 1=TR (rows 0,1; cols 2,3), 2=BL (rows 2,3; cols 0,1), 3=BR (rows 2,3; cols 2,3)
    # C_block: 0=left (cols 0,1), 1=right (cols 2,3)
    
    block_configs = [
        (0, 0, 0),  # A_left @ B_TL -> C_left
        (1, 2, 0),  # A_right @ B_BL -> C_left
        (0, 1, 1),  # A_left @ B_TR -> C_right
        (1, 3, 1),  # A_right @ B_BR -> C_right
    ]
    
    for (a_block, b_block, c_block) in block_configs:
        # A block: a_block=0 -> cols 0,1; a_block=1 -> cols 2,3
        a_col_offset = a_block * 2
        # B block: b_block 0=TL,1=TR,2=BL,3=BR
        # b_block = b_row * 2 + b_col where b_row=0 or 1, b_col=0 or 1
        b_row = b_block // 2
        b_col = b_block % 2
        b_row_offset = b_row * 2
        b_col_offset = b_col * 2
        # C block: c_block=0 -> cols 0,1; c_block=1 -> cols 2,3
        c_col_offset = c_block * 2
        
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            for local_i in range(2):
                for local_k in range(2):
                    # A element: row i (0 or 1), col = a_col_offset + local_k
                    i = local_i
                    k = a_col_offset + local_k
                    U[i * m + k][col] = u_coeffs[local_i * 2 + local_k]
            
            for local_k in range(2):
                for local_j in range(2):
                    # B element: row = b_row_offset + local_k, col = b_col_offset + local_j
                    k = b_row_offset + local_k
                    j = b_col_offset + local_j
                    V[k * p + j][col] = v_coeffs[local_k * 2 + local_j]
            
            for local_k in range(2):
                for local_i in range(2):
                    # C element: row i (0 or 1), col = c_col_offset + local_k
                    # W is indexed by [k*n + i] where k is col of C, i is row of C
                    i = local_i
                    k = c_col_offset + local_k
                    W[k * n + i][col] = w_coeffs[local_i * 2 + local_k]
            
            col += 1
    
    return U, V, W


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W


def construct():
    deadline = time.time() + 140.0  # safety margin
    
    # Try Strassen rank-28 decomposition
    U, V, W = _build_strassen_2x4x4()
    
    if time.time() < deadline - 5.0:
        if _verify_decomposition(U, V, W, n=2, m=4, p=4, num_trials=5, seed=42):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank-32
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
