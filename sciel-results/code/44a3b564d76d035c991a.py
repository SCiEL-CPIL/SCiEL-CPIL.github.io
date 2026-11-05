import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) as lists of lists.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B00)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n×m×p block."""
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
    return U, V, rank

def _block_decomp(n, m, p):
    """
    Get optimal decomposition for an n×m×p block.
    Uses Strassen for 2×2×2, schoolbook otherwise.
    """
    if n == 2 and m == 2 and p == 2:
        U, V = _strassen_2x2()
        return U, V, 7
    else:
        U, V, r = _schoolbook_block(n, m, p)
        return U, V, r

def _construct_221_partition():
    """
    Construct 5×5×5 matmul using 2+2+1 block partition.
    Blocks: rows split [0:2, 2:4, 4:5], cols same, inner same.
    For each (i_block, k_block, j_block), decompose the sub-matmul.
    """
    n, m, p = 5, 5, 5
    
    # Block boundaries: 0, 2, 4, 5
    boundaries = [0, 2, 4, 5]
    block_sizes = [2, 2, 1]
    
    # Precompute decompositions for each unique (n_i, m_j, p_k) combination
    decomp_cache = {}
    
    # Total rank
    total_rank = 0
    block_info = []  # (i_start, i_end, k_start, k_end, j_start, j_end, n_i, m_k, p_j, rank)
    
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                ni = block_sizes[bi]
                mk = block_sizes[bk]
                pj = block_sizes[bj]
                
                key = (ni, mk, pj)
                if key not in decomp_cache:
                    U, V, r = _block_decomp(ni, mk, pj)
                    decomp_cache[key] = (U, V, r)
                
                U, V, r = decomp_cache[key]
                total_rank += r
                
                i_start = boundaries[bi]
                i_end = boundaries[bi+1]
                k_start = boundaries[bk]
                k_end = boundaries[bk+1]
                j_start = boundaries[bj]
                j_end = boundaries[bj+1]
                
                block_info.append((i_start, i_end, k_start, k_end, j_start, j_end, ni, mk, pj, r))
    
    # Build global matrices
    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (i_start, i_end, k_start, k_end, j_start, j_end, ni, mk, pj, r) in block_info:
        key = (ni, mk, pj)
        U, V, r = decomp_cache[key]
        
        # Fill U_global: rows are (i, k) for A[i,k]
        for i in range(ni):
            for k in range(mk):
                for c in range(r):
                    U_global[(i_start + i) * m + (k_start + k)][col + c] = U[i * mk + k][c]
        
        # Fill V_global: rows are (k, j) for B[k,j]
        for k in range(mk):
            for j in range(pj):
                for c in range(r):
                    V_global[(k_start + k) * p + (j_start + j)][col + c] = V[k * pj + j][c]
        
        # Fill W_global: rows are (j, i) for C[i,j]
        for j in range(pj):
            for i in range(ni):
                for c in range(r):
                    W_global[(j_start + j) * n + (i_start + i)][col + c] = W[j * ni + i][c]
        
        col += r
    
    return {"U": U_global, "V": V_global, "W": W_global}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition is correct."""
    rank = len(U[0])
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    # Use numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            # f_r = sum_k U[i*m+k, r] * A[i,k]
            # g_r = sum_k V[k*p+j, r] * B[k,j]
            # C[i,j] = sum_r W[j*n+i, r] * f_r * g_r
            
            f = np.zeros(rank)
            g = np.zeros(rank)
            for k in range(m):
                f += A[i, k] * U_np[i*m+k, :]
                g += B[k, j] * V_np[k*p+j, :]
            
            C_computed[i, j] = np.dot(W_np[j*n+i, :], f * g)
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _try_32_partition():
    """
    Try a 3+2 partition: split 5 into 3+2 for all three dimensions.
    This gives blocks:
    - (3,3,3): schoolbook rank 27
    - (3,3,2): schoolbook rank 18
    - (3,2,3): schoolbook rank 18
    - (3,2,2): schoolbook rank 12
    - (2,3,3): schoolbook rank 18
    - (2,3,2): schoolbook rank 12
    - (2,2,3): schoolbook rank 12
    - (2,2,2): Strassen rank 7
    
    Total: 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124
    This is worse than 117, so we skip it unless something else works.
    Actually, let's compute more carefully:
    Blocks (ni, mk, pj):
    (3,3,3): 27
    (3,3,2): 18
    (3,2,3): 18
    (3,2,2): 12
    (2,3,3): 18
    (2,3,2): 12
    (2,2,3): 12
    (2,2,2): 7
    Sum = 27+18+18+12+18+12+12+7 = 124
    
    This is worse. Let's not pursue this.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try 2+2+1 partition with Strassen
    if time.time() < deadline:
        try:
            result = _construct_221_partition()
            rank = len(result["U"][0])
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
        except Exception as e:
            print(f"2+2+1 partition construction failed: {e}")
    
    return fallback

# EVOLVE-BLOCK-END
