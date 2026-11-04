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
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B11-B21)
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_225():
    """
    Construct a decomposition for (2,2,5) matrix multiplication.
    Uses Strassen for paired columns and schoolbook for the last column.
    Returns U (4, rank), V (10, rank), W (10, rank) with rank <= 18.
    """
    strassen_U, strassen_V, strassen_W = _strassen_222()
    total_rank = 18
    U = np.zeros((4, total_rank), dtype=np.int64)
    V = np.zeros((10, total_rank), dtype=np.int64)
    W = np.zeros((10, total_rank), dtype=np.int64)
    
    col_offset = 0
    # Pair (col 0, col 1)
    for c in range(7):
        U[:, col_offset + c] = strassen_U[:, c]
        for i in range(2):
            for j in range(2):
                sv = strassen_V[i*2+j, c]
                if sv != 0:
                    V[i*5+j, col_offset + c] += sv
        for i in range(2):
            for j in range(2):
                sw = strassen_W[i*2+j, c]
                if sw != 0:
                    W[i*5+j, col_offset + c] += sw
    col_offset += 7
    
    # Pair (col 2, col 3)
    for c in range(7):
        U[:, col_offset + c] = strassen_U[:, c]
        for i in range(2):
            for j in range(2):
                sv = strassen_V[i*2+j, c]
                if sv != 0:
                    V[i*5+(2+j), col_offset + c] += sv
        for i in range(2):
            for j in range(2):
                sw = strassen_W[i*2+j, c]
                if sw != 0:
                    W[i*5+(2+j), col_offset + c] += sw
    col_offset += 7
    
    # Column 4
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
    """
    rank = 10
    U = np.zeros((2, rank), dtype=np.int64)
    V = np.zeros((10, rank), dtype=np.int64)
    W = np.zeros((5, rank), dtype=np.int64)
    
    col = 0
    for k in range(5):
        U[0, col] = 1
        V[0*5+k, col] = 1
        W[k, col] = 1
        col += 1
        U[1, col] = 1
        V[1*5+k, col] = 1
        W[k, col] = 1
        col += 1
    
    return U, V, W, rank

def _construct_325():
    """
    Construct a decomposition for (3,2,5) matrix multiplication.
    """
    U_125, V_125, W_125, rank_125 = _construct_125()
    U_225, V_225, W_225, rank_225 = _construct_225()
    
    total_rank = rank_125 + rank_225
    U = np.zeros((6, total_rank), dtype=np.int64)
    V = np.zeros((10, total_rank), dtype=np.int64)
    W = np.zeros((15, total_rank), dtype=np.int64)
    
    U[0:2, 0:rank_125] = U_125
    V[0:10, 0:rank_125] = V_125
    W[0:5, 0:rank_125] = W_125
    
    offset = rank_125
    U[2:6, offset:offset+rank_225] = U_225
    V[0:10, offset:offset+rank_225] = V_225
    W[5:15, offset:offset+rank_225] = W_225
    
    return U, V, W, total_rank

def _construct_345():
    """
    Construct a decomposition for (3,4,5) matrix multiplication.
    """
    U_325, V_325, W_325, rank_325 = _construct_325()
    
    total_rank = 2 * rank_325
    U = np.zeros((12, total_rank), dtype=np.int64)
    V = np.zeros((20, total_rank), dtype=np.int64)
    W = np.zeros((15, total_rank), dtype=np.int64)
    
    U[0:6, 0:rank_325] = U_325
    V[0:10, 0:rank_325] = V_325
    W[0:15, 0:rank_325] = W_325
    
    offset = rank_325
    U[6:12, offset:offset+rank_325] = U_325
    V[10:20, offset:offset+rank_325] = V_325
    W[0:15, offset:offset+rank_325] = W_325
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 4, 5
    
    try:
        U_best, V_best, W_best, rank_best = _construct_345()
        
        if _verify_decomposition(U_best, V_best, W_best, n, m, p, rank_best, seed=42, num_tests=30):
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    except Exception:
        pass
    
    U_best, V_best, W_best, rank_best = _schoolbook_decomposition(n, m, p)
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
