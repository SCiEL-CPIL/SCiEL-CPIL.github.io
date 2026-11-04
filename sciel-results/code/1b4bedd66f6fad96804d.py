import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, rank, seed=42, num_tests=50):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randint(-100, 100, size=(n, m)).astype(np.int64)
        B = rng.randint(-100, 100, size=(m, p)).astype(np.int64)
        C_expected = A @ B
        
        C_recomputed = np.zeros((n, p), dtype=np.int64)
        for c in range(rank):
            ua = np.dot(U[:, c], A.ravel())
            vb = np.dot(V[:, c], B.ravel())
            Wc = W[:, c].reshape(p, n).T
            C_recomputed += Wc * (ua * vb)
        
        if not np.array_equal(C_recomputed, C_expected):
            return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _strassen_222():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c] corresponds to A[i,j], V[i*2+j, c] to B[i,j], W[i*2+j, c] to C[i,j]
    """
    # Strassen's 7 products for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[1, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B21
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22(B11-B21)
    U[3, 3] = 1  # A22
    V[0, 3] = 1  # B11
    V[1, 3] = -1 # B21
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[1, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[1, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _construct_225():
    """
    Construct a decomposition for (2,2,5) matrix multiplication.
    Uses Strassen for paired columns and schoolbook for the last column.
    Returns U (4, rank), V (10, rank), W (10, rank) with rank <= 18.
    """
    # B is 2x5, so we index B as B[j*5+k] for row j, col k
    # A is 2x2, indexed as A[i*2+j]
    # C is 2x5, indexed as C[i*5+j]
    
    # Strategy: Pair columns (0,1), (2,3), and handle column 4 separately
    # For each pair, use Strassen on 2x2x2
    # For column 4, use schoolbook 2x2x1 (rank 4)
    
    strassen_U, strassen_V, strassen_W = _strassen_222()
    
    # Total rank: 7 (pair 0,1) + 7 (pair 2,3) + 4 (col 4) = 18
    total_rank = 18
    U = np.zeros((4, total_rank), dtype=np.int64)
    V = np.zeros((10, total_rank), dtype=np.int64)
    W = np.zeros((10, total_rank), dtype=np.int64)
    
    col_offset = 0
    
    # Pair (col 0, col 1): use Strassen
    # B submatrix: rows 0-1, cols 0-1 -> indices 0,1,5,6 in V
    # C submatrix: rows 0-1, cols 0-1 -> indices 0,1,5,6 in W
    for c in range(7):
        # U: same as strassen_U
        U[:, col_offset + c] = strassen_U[:, c]
        # V: map strassen_V indices to B[0:2, 0:2]
        # strassen_V[i,j] -> V[i*2+j] but our B is 2x5, so B[i, j] is at index i*5+j
        # For cols 0,1: B[i,0] = V[i*5+0], B[i,1] = V[i*5+1]
        # strassen_V has indices: 0->B00, 1->B01, 2->B10, 3->B11
        # So V[i*5+j] where i is row, j is col within the pair
        for i in range(2):
            for j in range(2):
                sv = strassen_V[i*2+j, c]
                if sv != 0:
                    V[i*5+j, col_offset + c] += sv
        # W: map strassen_W indices to C[0:2, 0:2]
        for i in range(2):
            for j in range(2):
                sw = strassen_W[i*2+j, c]
                if sw != 0:
                    W[i*5+j, col_offset + c] += sw
    col_offset += 7
    
    # Pair (col 2, col 3): use Strassen
    for c in range(7):
        U[:, col_offset + c] = strassen_U[:, c]
        # B[i, 2] = V[i*5+2], B[i, 3] = V[i*5+3]
        for i in range(2):
            for j in range(2):
                sv = strassen_V[i*2+j, c]
                if sv != 0:
                    V[i*5+(2+j), col_offset + c] += sv
        # C[i, 2] = W[i*5+2], C[i, 3] = W[i*5+3]
        for i in range(2):
            for j in range(2):
                sw = strassen_W[i*2+j, c]
                if sw != 0:
                    W[i*5+(2+j), col_offset + c] += sw
    col_offset += 7
    
    # Column 4: schoolbook 2x2x1, rank 4
    for i in range(2):
        for j in range(2):
            U[i*2+j, col_offset] = 1
            V[j*5+4, col_offset] = 1
            W[i*5+4, col_offset] = 1
            col_offset += 1
    
    return U, V, W, total_rank

def _construct_125():
    """
    Construct a decomposition for (1,2,5) matrix multiplication.
    A is 1x2, B is 2x5, C is 1x5.
    Rank is 10 (each output depends on 2 inputs).
    Returns U (2, 10), V (10, 10), W (5, 10).
    """
    # C[k] = A[0] * B[0,k] + A[1] * B[1,k] for k=0..4
    # We can write this as: for each k, C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    # This is rank 2 per output, total rank 10
    
    rank = 10
    U = np.zeros((2, rank), dtype=np.int64)
    V = np.zeros((10, rank), dtype=np.int64)
    W = np.zeros((5, rank), dtype=np.int64)
    
    col = 0
    for k in range(5):
        # Term 1: A[0] * B[0,k]
        U[0, col] = 1
        V[0*5+k, col] = 1
        W[k, col] = 1
        col += 1
        # Term 2: A[1] * B[1,k]
        U[1, col] = 1
        V[1*5+k, col] = 1
        W[k, col] = 1
        col += 1
    
    return U, V, W, rank

def _construct_325():
    """
    Construct a decomposition for (3,2,5) matrix multiplication.
    A is 3x2, B is 2x5, C is 3x5.
    Split A into A_top (1x2) and A_bot (2x2).
    C = A_top @ B + A_bot @ B
    Returns U (6, rank), V (10, rank), W (15, rank).
    """
    U_125, V_125, W_125, rank_125 = _construct_125()
    U_225, V_225, W_225, rank_225 = _construct_225()
    
    total_rank = rank_125 + rank_225  # 10 + 18 = 28
    U = np.zeros((6, total_rank), dtype=np.int64)
    V = np.zeros((10, total_rank), dtype=np.int64)
    W = np.zeros((15, total_rank), dtype=np.int64)
    
    # Part 1: A_top (1x2) @ B (2x5) -> C_top (1x5)
    # A_top indices: 0,1 (A[0,0], A[0,1])
    # C_top indices: 0..4 (C[0,0]..C[0,4])
    U[0:2, 0:rank_125] = U_125
    V[0:10, 0:rank_125] = V_125
    W[0:5, 0:rank_125] = W_125  # C[0, k] for k=0..4
    
    # Part 2: A_bot (2x2) @ B (2x5) -> C_bot (2x5)
    # A_bot indices: 2,3,4,5 (A[1,0], A[1,1], A[2,0], A[2,1])
    # C_bot indices: 5..14 (C[1,0]..C[1,4], C[2,0]..C[2,4])
    offset = rank_125
    U[2:6, offset:offset+rank_225] = U_225
    V[0:10, offset:offset+rank_225] = V_225
    # W_225 has shape (10, rank_225) for C[0:2, 0:5]
    # We need to map to C[1:3, 0:5] which is indices 5..14
    W[5:15, offset:offset+rank_225] = W_225
    
    return U, V, W, total_rank

def _construct_345():
    """
    Construct a decomposition for (3,4,5) matrix multiplication.
    Split A (3x4) into A_left (3x2) and A_right (3x2).
    Split B (4x5) into B_top (2x5) and B_bot (2x5).
    C = A_left @ B_top + A_right @ B_bot
    Each part is (3,2,5).
    Returns U (12, rank), V (20, rank), W (15, rank).
    """
    U_325, V_325, W_325, rank_325 = _construct_325()
    
    total_rank = 2 * rank_325  # 2 * 28 = 56
    U = np.zeros((12, total_rank), dtype=np.int64)
    V = np.zeros((20, total_rank), dtype=np.int64)
    W = np.zeros((15, total_rank), dtype=np.int64)
    
    # Part 1: A_left (3x2) @ B_top (2x5)
    # A_left: rows 0-2, cols 0-1 of A -> indices 0..5 (A[i,0], A[i,1])
    # B_top: rows 0-1, cols 0-4 of B -> indices 0..9 (B[j,k])
    # C: all 15 entries
    U[0:6, 0:rank_325] = U_325
    V[0:10, 0:rank_325] = V_325
    W[0:15, 0:rank_325] = W_325
    
    # Part 2: A_right (3x2) @ B_bot (2x5)
    # A_right: rows 0-2, cols 2-3 of A -> indices 6..11 (A[i,2], A[i,3])
    # B_bot: rows 2-3, cols 0-4 of B -> indices 10..19 (B[j,k])
    offset = rank_325
    U[6:12, offset:offset+rank_325] = U_325
    V[10:20, offset:offset+rank_325] = V_325
    W[0:15, offset:offset+rank_325] = W_325
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 5
    
    # Try the constructive Strassen-based decomposition (rank 56)
    try:
        U_best, V_best, W_best, rank_best = _construct_345()
        
        # Verify the decomposition
        if _verify_decomposition(U_best, V_best, W_best, n, m, p, rank_best, seed=42, num_tests=30):
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook decomposition (rank 60)
    U_best, V_best, W_best, rank_best = _schoolbook_decomposition(n, m, p)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
