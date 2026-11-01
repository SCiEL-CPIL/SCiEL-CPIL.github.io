import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    max_err = max(max_err, abs(val - 1.0))
    return max_err

def _build_strassen_28():
    """
    Build Strassen-based rank-28 decomposition for 2x4 * 4x4.
    
    We view the 2x4 matrix A as two 2x2 blocks: A = [A_left | A_right]
    and the 4x4 matrix B as four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    
    The product C = A * B is 2x4, which we view as two 2x2 blocks: C = [C_left | C_right]
    
    C_left = A_left * B00 + A_right * B10
    C_right = A_left * B01 + A_right * B11
    
    We apply Strassen to each of the two 2x2*2x2 multiplications,
    but we can share some intermediate products.
    
    Actually, let's think more carefully. We have:
    - A is 2x4: rows i=0,1; columns j=0,1,2,3
    - B is 4x4: rows j=0,1,2,3; columns k=0,1,2,3
    - C is 2x4: rows i=0,1; columns k=0,1,2,3
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's partition:
    - A = [A0 | A1] where A0 is 2x2 (cols 0,1) and A1 is 2x2 (cols 2,3)
    - B = [[B0 | B1], [B2 | B3]] where each Bx is 2x2
      B0 = B[0:2, 0:2], B1 = B[0:2, 2:4]
      B2 = B[2:4, 0:2], B3 = B[2:4, 2:4]
    
    C = [C0 | C1] where C0 = 2x2 (cols 0,1) and C1 = 2x2 (cols 2,3)
    
    C0 = A0 * B0 + A1 * B2
    C1 = A0 * B1 + A1 * B3
    
    Each of these is a sum of two 2x2*2x2 products.
    We can use Strassen for each 2x2*2x2 product, giving 7 multiplications each,
    so 14 total for C0 and 14 for C1 = 28 total.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    def strassen_2x2_mult(A_rows, B_rows, C_rows, col_offset):
        """
        Apply Strassen to compute A(2x2) * B(2x2) -> C(2x2).
        
        A_rows: [A00, A01, A10, A11] -> indices in U
        B_rows: [B00, B01, B10, B11] -> indices in V  
        C_rows: [C00, C01, C10, C11] -> indices in W
        
        Returns list of (matrix, row, col, val) tuples to add.
        """
        entries = []
        
        # M1 = (A00+A11)*(B00+B11)
        entries.append(('U', A_rows[0], col_offset, 1))
        entries.append(('U', A_rows[3], col_offset, 1))
        entries.append(('V', B_rows[0], col_offset, 1))
        entries.append(('V', B_rows[3], col_offset, 1))
        entries.append(('W', C_rows[0], col_offset, 1))
        entries.append(('W', C_rows[1], col_offset, -1))
        entries.append(('W', C_rows[2], col_offset, -1))
        entries.append(('W', C_rows[3], col_offset, 1))
        
        # M2 = (A10+A11)*B00
        col2 = col_offset + 1
        entries.append(('U', A_rows[2], col2, 1))
        entries.append(('U', A_rows[3], col2, 1))
        entries.append(('V', B_rows[0], col2, 1))
        entries.append(('W', C_rows[2], col2, 1))
        entries.append(('W', C_rows[3], col2, -1))
        
        # M3 = A00*(B10-B11)
        col3 = col_offset + 2
        entries.append(('U', A_rows[0], col3, 1))
        entries.append(('V', B_rows[2], col3, 1))
        entries.append(('V', B_rows[3], col3, -1))
        entries.append(('W', C_rows[0], col3, 1))
        entries.append(('W', C_rows[2], col3, 1))
        
        # M4 = A11*(B01-B10)
        col4 = col_offset + 3
        entries.append(('U', A_rows[3], col4, 1))
        entries.append(('V', B_rows[1], col4, 1))
        entries.append(('V', B_rows[2], col4, -1))
        entries.append(('W', C_rows[1], col4, 1))
        entries.append(('W', C_rows[3], col4, 1))
        
        # M5 = (A00+A10)*B11
        col5 = col_offset + 4
        entries.append(('U', A_rows[0], col5, 1))
        entries.append(('U', A_rows[2], col5, 1))
        entries.append(('V', B_rows[3], col5, 1))
        entries.append(('W', C_rows[2], col5, 1))
        entries.append(('W', C_rows[3], col5, -1))
        
        # M6 = (A01-A10)*(B00+B10)
        col6 = col_offset + 5
        entries.append(('U', A_rows[1], col6, 1))
        entries.append(('U', A_rows[2], col6, -1))
        entries.append(('V', B_rows[0], col6, 1))
        entries.append(('V', B_rows[2], col6, 1))
        entries.append(('W', C_rows[0], col6, -1))
        entries.append(('W', C_rows[1], col6, 1))
        entries.append(('W', C_rows[2], col6, 1))
        entries.append(('W', C_rows[3], col6, -1))
        
        # M7 = (A01-A11)*(B01+B11)
        col7 = col_offset + 6
        entries.append(('U', A_rows[1], col7, 1))
        entries.append(('U', A_rows[3], col7, -1))
        entries.append(('V', B_rows[1], col7, 1))
        entries.append(('V', B_rows[3], col7, 1))
        entries.append(('W', C_rows[0], col7, 1))
        entries.append(('W', C_rows[1], col7, 1))
        entries.append(('W', C_rows[2], col7, -1))
        entries.append(('W', C_rows[3], col7, -1))
        
        return entries
    
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # U index: i*4 + j
    # A0 (left 2x2): A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,4,5
    # A1 (right 2x2): A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2,3,6,7
    
    # B is 4x4: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}
    # V index: j*4 + k
    # B0 = B[0:2, 0:2]: B[0,0],B[0,1],B[1,0],B[1,1] -> V indices 0,1,4,5
    # B1 = B[0:2, 2:4]: B[0,2],B[0,3],B[1,2],B[1,3] -> V indices 2,3,6,7
    # B2 = B[2:4, 0:2]: B[2,0],B[2,1],B[3,0],B[3,1] -> V indices 8,9,12,13
    # B3 = B[2:4, 2:4]: B[2,2],B[2,3],B[3,2],B[3,3] -> V indices 10,11,14,15
    
    # C is 2x4: C[i,k] for i in {0,1}, k in {0,1,2,3}
    # W index: k*2 + i (trace convention W[k*n+i] with n=2)
    # C0 (left 2x2): C[0,0],C[0,1],C[1,0],C[1,1] -> W indices 0,1,2,3
    # C1 (right 2x2): C[0,2],C[0,3],C[1,2],C[1,3] -> W indices 4,5,6,7
    
    # C0 = A0 * B0 + A1 * B2
    # Strassen for A0*B0:
    A0 = [0, 1, 4, 5]  # A00, A01, A10, A11
    B0 = [0, 1, 4, 5]  # B00, B01, B10, B11
    C0 = [0, 1, 2, 3]  # C00, C01, C10, C11
    entries = strassen_2x2_mult(A0, B0, C0, 0)
    for mat, row, col, val in entries:
        if mat == 'U':
            U[row][col] += val
        elif mat == 'V':
            V[row][col] += val
        elif mat == 'W':
            W[row][col] += val
    
    # Strassen for A1*B2:
    A1 = [2, 3, 6, 7]  # A[0,2], A[0,3], A[1,2], A[1,3]
    B2 = [8, 9, 12, 13]  # B[2,0], B[2,1], B[3,0], B[3,1]
    C0 = [0, 1, 2, 3]  # same C0
    entries = strassen_2x2_mult(A1, B2, C0, 7)
    for mat, row, col, val in entries:
        if mat == 'U':
            U[row][col] += val
        elif mat == 'V':
            V[row][col] += val
        elif mat == 'W':
            W[row][col] += val
    
    # C1 = A0 * B1 + A1 * B3
    # Strassen for A0*B1:
    B1 = [2, 3, 6, 7]  # B[0,2], B[0,3], B[1,2], B[1,3]
    C1 = [4, 5, 6, 7]  # C[0,2], C[0,3], C[1,2], C[1,3]
    entries = strassen_2x2_mult(A0, B1, C1, 14)
    for mat, row, col, val in entries:
        if mat == 'U':
            U[row][col] += val
        elif mat == 'V':
            V[row][col] += val
        elif mat == 'W':
            W[row][col] += val
    
    # Strassen for A1*B3:
    B3 = [10, 11, 14, 15]  # B[2,2], B[2,3], B[3,2], B[3,3]
    C1 = [4, 5, 6, 7]  # same C1
    entries = strassen_2x2_mult(A1, B3, C1, 21)
    for mat, row, col, val in entries:
        if mat == 'U':
            U[row][col] += val
        elif mat == 'V':
            V[row][col] += val
        elif mat == 'W':
            W[row][col] += val
    
    return U, V, W

def _build_schoolbook_32():
    """Fallback: schoolbook rank-32 decomposition."""
    n, m, p = 2, 4, 4
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
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Try Strassen rank 28
    U, V, W = _build_strassen_28()
    err = _verify_decomposition(U, V, W)
    
    if err < 1e-9:
        return {"U": U, "V": V, "W": W}
    
    # If Strassen verification fails, fall back to schoolbook
    U, V, W = _build_schoolbook_32()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
