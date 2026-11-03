"""Tensor decomposition for 3x4x5 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = s + U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _construct_strassen_blocks():
    """
    Construct a rank-56 decomposition using:
    - 4 Strassen 2x2 blocks (rank 7 each = 28 total) for (i in {1,2}, j in {0,1} or {2,3}, k in {0,1} or {3,4})
    - 1 schoolbook 1x4x5 block (rank 20) for i=0
    - 2 schoolbook 2x2x1 blocks (rank 4 each = 8 total) for (i in {1,2}, j in {0,1} or {2,3}, k=2)
    
    Total: 28 + 20 + 8 = 56
    """
    n, m, p = 3, 4, 5
    
    U = [[0]*56 for _ in range(n*m)]  # 12 rows x 56 cols
    V = [[0]*56 for _ in range(m*p)]  # 20 rows x 56 cols
    W = [[0]*56 for _ in range(p*n)]  # 15 rows x 56 cols
    
    col = 0
    
    # Strassen 2x2x2 decomposition values
    # For 2x2 matrix mult: C[0,0]=M1-M3+M4+M5, etc.
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B11), M4=A11(B01-B00),
    # M5=(A00+A01)B11, M6=(A01-A10)(B00+B10), M7=(A10-A11)(B01+B11)
    # C[0,0] = M1 - M3 + M4 + M5
    # C[0,1] = M2 + M5
    # C[1,0] = M3 + M4
    # C[1,1] = M1 - M2 + M6 + M7
    
    # Strassen terms: (U_coeffs, V_coeffs, W_coeffs) for each of 7 terms
    # U is 2x2 (for i_local, j_local), V is 2x2 (for j_local, k_local), W is 2x2 (for k_local, i_local)
    strassen_terms = [
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: [[1,0],[0,1]], V: [[1,0],[1,1]], W: [[1,0],[0,1]]  (W[k,i]: W[0,0]=1, W[1,1]=1)
        ([[1,0],[0,1]], [[1,0],[1,1]], [[1,0],[0,1]]),
        # Term 2: M2 = (A10+A11)B00
        # U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[0,1],[0,0]]  (contributes to C[1,0] with +1... wait)
        # Actually C[0,1] = M2 + M5, so W should map to C[0,1] which is (i=0, k=1)
        # W[k,i]: W[1,0] = 1 means k=1, i=0 -> C[0,1]
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        # Term 3: M3 = A00(B10-B11)
        # U: [[1,0],[0,0]], V: [[0,0],[1,-1]], W: [[0,0],[1,1]]  (contributes to C[0,0] with -1, C[1,0] with +1)
        # C[0,0] gets -M3, C[1,0] gets +M3
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[-1,0],[0,1]]),
        # Term 4: M4 = A11(B01-B00)
        # U: [[0,0],[0,1]], V: [[-1,1],[0,0]], W: [[1,0],[0,1]]  (contributes to C[0,0] with +1, C[1,0] with +1)
        ([[0,0],[0,1]], [[-1,1],[0,0]], [[1,0],[0,1]]),
        # Term 5: M5 = (A00+A01)B11
        # U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[1,0],[0,0]]  (contributes to C[0,0] with +1, C[0,1] with +1)
        ([[1,1],[0,0]], [[0,0],[0,1]], [[1,0],[1,0]]),
        # Term 6: M6 = (A01-A10)(B00+B10)
        # U: [[0,1],[-1,0]], V: [[1,1],[0,0]], W: [[0,0],[0,1]]  (contributes to C[1,1] with +1)
        ([[0,1],[-1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # Term 7: M7 = (A10-A11)(B01+B11)
        # U: [[0,0],[1,-1]], V: [[0,1],[0,1]], W: [[0,0],[0,1]]  (contributes to C[1,1] with +1)
        ([[0,0],[1,-1]], [[0,1],[0,1]], [[0,0],[0,1]]),
    ]
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add a Strassen 2x2x2 block at offset (i_off, j_off, k_off). Returns columns used."""
        nonlocal col
        used = []
        for t in range(7):
            u_mat, v_mat, w_mat = strassen_terms[t]
            for il in range(2):
                for jl in range(2):
                    U[(i_off+il)*m + (j_off+jl)][col] = u_mat[il][jl]
            for jl in range(2):
                for kl in range(2):
                    V[(j_off+jl)*p + (k_off+kl)][col] = v_mat[jl][kl]
            for kl in range(2):
                for il in range(2):
                    W[(k_off+kl)*n + (i_off+il)][col] = w_mat[kl][il]
            used.append(col)
            col += 1
        return used
    
    def add_schoolbook_block(i_off, j_off, k_off, ni, nj, nk):
        """Add a schoolbook block of size ni x nj x nk at offset. Returns columns used."""
        nonlocal col
        used = []
        for il in range(ni):
            for jl in range(nj):
                for kl in range(nk):
                    U[(i_off+il)*m + (j_off+jl)][col] = 1
                    V[(j_off+jl)*p + (k_off+kl)][col] = 1
                    W[(k_off+kl)*n + (i_off+il)][col] = 1
                    used.append(col)
                    col += 1
        return used
    
    # Block 1: i=0, j=0..3, k=0..4 -> schoolbook 1x4x5 (rank 20)
    add_schoolbook_block(0, 0, 0, 1, 4, 5)
    
    # Block 2: i=1..2, j=0..1, k=0..1 -> Strassen (rank 7)
    add_strassen_block(1, 0, 0)
    
    # Block 3: i=1..2, j=0..1, k=2 -> schoolbook 2x2x1 (rank 4)
    add_schoolbook_block(1, 0, 2, 2, 2, 1)
    
    # Block 4: i=1..2, j=0..1, k=3..4 -> Strassen (rank 7)
    add_strassen_block(1, 0, 3)
    
    # Block 5: i=1..2, j=2..3, k=0..1 -> Strassen (rank 7)
    add_strassen_block(1, 2, 0)
    
    # Block 6: i=1..2, j=2..3, k=2 -> schoolbook 2x2x1 (rank 4)
    add_schoolbook_block(1, 2, 2, 2, 2, 1)
    
    # Block 7: i=1..2, j=2..3, k=3..4 -> Strassen (rank 7)
    add_strassen_block(1, 2, 3)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Strassen block construction failed verification")
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Build the rank-56 decomposition
    result = _construct_strassen_blocks()
    
    # Verify one more time
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return result

# EVOLVE-BLOCK-END
