"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that U,V,W correctly decompose matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_rank_23_strassen_block():
    """
    Construct a rank-23 decomposition for 3x3 matrix multiplication by
    partitioning into 2x2 and 1x1 blocks, applying Strassen's 7-term algorithm
    to the 2x2 sub-blocks, and handling cross-terms with the 1x1 blocks.
    
    Matrix partition:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]
    
    C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    
    The 2x2 top-left block uses Strassen (7 multiplications).
    Cross terms with the 3rd row/column use direct multiplication.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 9 rows
    V = [[0] * rank for _ in range(m * p)]  # 9 rows
    W = [[0] * rank for _ in range(p * n)]  # 9 rows
    
    col = 0
    
    # Strassen's 7 terms for the 2x2 top-left block
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For Strassen term t, we need:
    # U factors: linear combination of A[i,j] entries
    # V factors: linear combination of B[j,k] entries
    # W factors: how this contributes to C[i,k]
    
    # Strassen term 1: M1 = (A11+A22)(B11+B22)
    # U: A[0,0]=idx0, A[1,1]=idx4
    # V: B[0,0]=idx0, B[1,1]=idx4
    # W: contributes to C11 with +1, C22 with +1
    # C11 is at W[0*3+0]=W[0], C22 is at W[1*3+1]=W[4]
    
    # Let's define helper to add a Strassen term
    def add_strassen_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs, col_idx):
        """Add one rank-1 term to the decomposition."""
        for idx, coeff in zip(u_indices, u_coeffs):
            if idx is not None and coeff != 0:
                U[idx][col_idx] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            if idx is not None and coeff != 0:
                V[idx][col_idx] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            if idx is not None and coeff != 0:
                W[idx][col_idx] += coeff
    
    # Strassen term 1: M1 = (A11+A22)(B11+B22)
    # U: A[0,0]=idx0, A[1,1]=idx4
    # V: B[0,0]=idx0, B[1,1]=idx4
    # W: C[0,0]=idx0 (+1), C[1,1]=idx4 (+1)
    add_strassen_term([0, 4], [1, 1], [0, 4], [1, 1], [0, 4], [1, 1], col)
    col += 1
    
    # Strassen term 2: M2 = (A21+A22)B11
    # U: A[1,0]=idx3, A[1,1]=idx4
    # V: B[0,0]=idx0
    # W: C[1,0]=idx3 (+1), C[1,1]=idx4 (-1)
    add_strassen_term([3, 4], [1, 1], [0], [1], [3, 4], [1, -1], col)
    col += 1
    
    # Strassen term 3: M3 = A11(B21-B22)
    # U: A[0,0]=idx0
    # V: B[1,0]=idx3, B[1,1]=idx4
    # W: C[0,1]=idx1 (+1), C[1,1]=idx4 (+1)
    add_strassen_term([0], [1], [3, 4], [1, -1], [1, 4], [1, 1], col)
    col += 1
    
    # Strassen term 4: M4 = A22(B12-B11)
    # U: A[1,1]=idx4
    # V: B[0,1]=idx1, B[0,0]=idx0
    # W: C[0,0]=idx0 (-1), C[1,0]=idx3 (+1)
    add_strassen_term([4], [1], [1, 0], [1, -1], [0, 3], [-1, 1], col)
    col += 1
    
    # Strassen term 5: M5 = (A11+A12)B22
    # U: A[0,0]=idx0, A[0,1]=idx1
    # V: B[1,1]=idx4
    # W: C[0,0]=idx0 (+1), C[0,1]=idx1 (+1)
    add_strassen_term([0, 1], [1, 1], [4], [1], [0, 1], [1, 1], col)
    col += 1
    
    # Strassen term 6: M6 = (A21-A11)(B11+B12)
    # U: A[1,0]=idx3, A[0,0]=idx0
    # V: B[0,0]=idx0, B[0,1]=idx1
    # W: C[1,1]=idx4 (+1)
    add_strassen_term([3, 0], [1, -1], [0, 1], [1, 1], [4], [1], col)
    col += 1
    
    # Strassen term 7: M7 = (A12-A22)(B21+B22)
    # U: A[0,1]=idx1, A[1,1]=idx4
    # V: B[1,0]=idx3, B[1,1]=idx4
    # W: C[0,0]=idx0 (+1), C[0,1]=idx1 (-1)
    add_strassen_term([1, 4], [1, -1], [3, 4], [1, 1], [0, 1], [1, -1], col)
    col += 1
    
    # Now handle cross terms with the 3rd row/column
    # We need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # That's 5 more terms, total 7+5 = 12... wait, let me recount.
    # Actually for rank 23, we need more terms. Let me reconsider.
    
    # The standard approach for 3x3 with Strassen on 2x2 block:
    # The 2x2x2 part takes 7 multiplications.
    # The cross terms:
    # Row 3 of C (C[2,*]): needs A[2,0], A[2,1], A[2,2] times B columns
    # Col 3 of C (C[*,2]): needs A rows times B[*,2]
    # But C[2,2] is counted once.
    
    # Let me just use direct multiplication for all cross terms:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # Each of these is a sum of 3 products, so we can do them directly.
    # But that would be 5*3 = 15 more terms, total 7+15 = 22. Hmm.
    
    # Actually, we can group some. Let me just add each product directly.
    
    # C[0,2] terms:
    # A[0,0]*B[0,2]: U[0], V[2], W[2]
    add_strassen_term([0], [1], [2], [1], [2], [1], col)
    col += 1
    # A[0,1]*B[1,2]: U[1], V[5], W[2]
    add_strassen_term([1], [1], [5], [1], [2], [1], col)
    col += 1
    # A[0,2]*B[2,2]: U[2], V[8], W[2]
    add_strassen_term([2], [1], [8], [1], [2], [1], col)
    col += 1
    
    # C[1,2] terms:
    # A[1,0]*B[0,2]: U[3], V[2], W[5]
    add_strassen_term([3], [1], [2], [1], [5], [1], col)
    col += 1
    # A[1,1]*B[1,2]: U[4], V[5], W[5]
    add_strassen_term([4], [1], [5], [1], [5], [1], col)
    col += 1
    # A[1,2]*B[2,2]: U[5], V[8], W[5]
    add_strassen_term([5], [1], [8], [1], [5], [1], col)
    col += 1
    
    # C[2,0] terms:
    # A[2,0]*B[0,0]: U[6], V[0], W[6]
    add_strassen_term([6], [1], [0], [1], [6], [1], col)
    col += 1
    # A[2,1]*B[1,0]: U[7], V[3], W[6]
    add_strassen_term([7], [1], [3], [1], [6], [1], col)
    col += 1
    # A[2,2]*B[2,0]: U[8], V[6], W[6]
    add_strassen_term([8], [1], [6], [1], [6], [1], col)
    col += 1
    
    # C[2,1] terms:
    # A[2,0]*B[0,1]: U[6], V[1], W[7]
    add_strassen_term([6], [1], [1], [1], [7], [1], col)
    col += 1
    # A[2,1]*B[1,1]: U[7], V[4], W[7]
    add_strassen_term([7], [1], [4], [1], [7], [1], col)
    col += 1
    # A[2,2]*B[2,1]: U[8], V[7], W[7]
    add_strassen_term([8], [1], [7], [1], [7], [1], col)
    col += 1
    
    # C[2,2] terms:
    # A[2,0]*B[0,2]: U[6], V[2], W[8]
    add_strassen_term([6], [1], [2], [1], [8], [1], col)
    col += 1
    # A[2,1]*B[1,2]: U[7], V[5], W[8]
    add_strassen_term([7], [1], [5], [1], [8], [1], col)
    col += 1
    # A[2,2]*B[2,2]: U[8], V[8], W[8]
    add_strassen_term([8], [1], [8], [1], [8], [1], col)
    col += 1
    
    # We've used 7 + 15 = 22 columns. We have rank 23, so one more.
    # Let me verify what we have so far.
    
    # Actually, I realize the Strassen decomposition above might not be correct
    # because the cross terms interact with the 2x2 block computation.
    # Let me verify and fix if needed.
    
    if col <= rank:
        # Pad with zeros if needed
        while col < rank:
            col += 1
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook():
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
    return {"U": U, "V": V, "W": W}

def construct():
    overall_deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try rank-23 Strassen-block decomposition
    start = time.time()
    r23 = _construct_rank_23_strassen_block()
    if r23 is not None:
        return r23
    
    if time.time() > overall_deadline:
        return _construct_schoolbook()
    
    # Fall back to schoolbook
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
