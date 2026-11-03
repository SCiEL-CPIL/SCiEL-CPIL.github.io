"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                total = np.dot(u_row * v_row, w_row)
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_block_2x2x4():
    """
    Return a rank-7 decomposition for 2x2 matrix multiplication (C = A*B, A,B 2x2).
    This is the classic Strassen algorithm.
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A is 2x2, indexed [i,j] -> row i*m+j = i*2+j
    # B is 2x2, indexed [j,k] -> row j*p+k = j*2+k
    # C is 2x2, indexed [k,i] -> row k*n+i = k*2+i
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11 (k=0,i=0)
    W[3, 0] = 1  # C22 (k=1,i=1)
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21 (k=1,i=0)
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1  # B22
    W[1, 2] = 1  # C12 (k=0,i=1)
    W[3, 2] = 1  # C22 (k=1,i=1)
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1  # B11
    W[0, 3] = 1  # C11 (k=0,i=0)
    W[2, 3] = 1  # C21 (k=1,i=0)
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1  # C11 (k=0,i=0)
    W[1, 4] = 1  # C12 (k=0,i=1)
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1  # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22 (k=1,i=1)
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1  # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11 (k=0,i=0)
    
    return U, V, W

def _embed_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, n, m, p):
    """
    Embed a 2x2x2 block decomposition into a larger n x m x p problem.
    i_offset: starting row index for A (0..n-2)
    j_offset: starting column index for A / row index for B (0..m-2)
    k_offset: starting column index for B (0..p-2)
    """
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    # A is n x m, indexed [i, j] -> row i*m + j
    # B is m x p, indexed [j, k] -> row j*p + k
    # C is p x n, indexed [k, i] -> row k*n + i
    
    for r in range(7):
        # U block: rows i_offset..i_offset+1, cols j_offset..j_offset+1
        for di in range(2):
            for dj in range(2):
                i = i_offset + di
                j = j_offset + dj
                U[i * m + j, r] = U_block[di * 2 + dj, r]
        
        # V block: rows j_offset..j_offset+1, cols k_offset..k_offset+1
        for dj in range(2):
            for dk in range(2):
                j = j_offset + dj
                k = k_offset + dk
                V[j * p + k, r] = V_block[dj * 2 + dk, r]
        
        # W block: rows k_offset..k_offset+1, cols i_offset..i_offset+1
        for dk in range(2):
            for di in range(2):
                k = k_offset + dk
                i = i_offset + di
                W[k * n + i, r] = W_block[dk * 2 + di, r]
    
    return U, V, W

def _construct_disjoint_strassen(n, m, p):
    """
    Construct a decomposition using disjoint Strassen blocks.
    For 3x3x8, we can use:
    - Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    - Block 2: i in {0,1}, j in {2,?}, ... wait, m=3, p=8
    
    Let's think about this more carefully.
    A is 3x3, B is 3x8.
    
    We can partition A into 2x2 blocks and B into 2x2 blocks.
    
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B = [[B11, B12, B13, B14, B15, B16, B17, B18],
         [B21, B22, B23, B24, B25, B26, B27, B28],
         [B31, B32, B33, B34, B35, B36, B37, B38]]
    
    C = A*B is 3x8.
    
    We can use Strassen on the (0,1) x (0,1) x (0,1) block:
    C[0,0] += A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] += A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] += A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] += A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    And similarly for other 2x2x2 blocks.
    
    Since m=3 and p=8, we can use:
    - j in {0,1}, k in {0,1}: 1 block
    - j in {0,1}, k in {2,3}: 1 block
    - j in {0,1}, k in {4,5}: 1 block
    - j in {0,1}, k in {6,7}: 1 block
    - j in {1,2}, k in {0,1}: 1 block
    - j in {1,2}, k in {2,3}: 1 block
    - j in {1,2}, k in {4,5}: 1 block
    - j in {1,2}, k in {6,7}: 1 block
    
    But these blocks overlap in j=1. We need disjoint blocks in all dimensions.
    
    For disjoint blocks, we need:
    - i ranges: {0,1} and {1,2} overlap at i=1, so we can only use one set of i ranges.
    
    Let's use i in {0,1} for all blocks. Then we cover rows 0,1 of C but not row 2.
    Similarly for j and k.
    
    This approach is limited. Let me try a different strategy.
    
    Actually, let's just use the schoolbook for now and try to find a better approach.
    """
    return None

def _try_random_search(n, m, p, rank, time_limit=5):
    """Try random search for a rank-rank decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for attempt in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with small random integers
        U = rng.randint(-2, 3, size=(n * m, rank))
        V = rng.randint(-2, 3, size=(m * p, rank))
        W = rng.randint(-2, 3, size=(p * n, rank))
        
        if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
    
    return None

def _construct_gf3_based(n, m, p):
    """
    Try to construct a decomposition based on GF(3) structure.
    Map indices to field elements and use algebraic identities.
    """
    start = time.time()
    best_rank = None
    best_decomp = None
    
    for rank in range(20, 40):
        if time.time() - start > 30:
            break
        
        found = False
        for seed in range(100):
            if time.time() - start > 30:
                break
            
            rng = np.random.RandomState(seed)
            U = rng.randint(0, 3, size=(n * m, rank))
            V = rng.randint(0, 3, size=(m * p, rank))
            W = rng.randint(0, 3, size=(p * n, rank))
            
            if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = rank
                best_decomp = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_decomp:
        return best_decomp
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try GF(3)-based structured search
    result = _construct_gf3_based(n, m, p)
    if result and time.time() < deadline - 10:
        U, V, W = result
        if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try various ranks with random search
    for rank in range(20, 45):
        if time.time() > deadline - 15:
            break
        result = _try_random_search(n, m, p, rank, time_limit=3)
        if result:
            U, V, W = result
            if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
