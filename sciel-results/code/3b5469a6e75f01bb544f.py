import time
import sys

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    if time.time() > deadline:
        return True
    return False

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_block_diagonal(n, m, p, block_decomp):
    """
    Build a block-diagonal decomposition from a block-level decomposition.
    block_decomp: list of tuples (U_block, V_block, W_block, i_range, j_range, k_range)
    where U_block is (block_size, rank_block), etc.
    """
    rank = 0
    for (_, _, _, _, _, rb) in block_decomp:
        rank += rb
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for U_block, V_block, W_block, i_range, j_range, k_range, rb in block_decomp:
        for r in range(rb):
            c = col_offset + r
            for li in range(i_range):
                gi = 2 * (j_range // 2) + 0  # simplified; this is a placeholder
                # Actually let's do this properly with offsets
                pass
        col_offset += rb
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _build_2x2x2_strassen():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where U is (4, 7), V is (4, 7), W is (4, 7).
    U maps input A (2x2, flattened as [a00, a01, a10, a11]) to 7 intermediate values.
    V maps input B (2x2, flattened as [b00, b01, b10, b11]) to 7 intermediate values.
    W maps 7 intermediate products to output C (2x2, flattened as [c00, c01, c10, c11]).
    
    Strassen's 7 multiplications:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)b00
    M3 = a00(b01 - b11)
    M4 = a11(b10 - b00)
    M5 = (a00 + a01)b11
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    Output:
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    """
    # U: 2x2 matrix -> 7 values
    # Flattened A: [a00, a01, a10, a11] (row-major)
    # M1 = (a00+a11)(b00+b11): U_M1 = [1, 0, 0, 1]
    # M2 = (a10+a11)b00: U_M2 = [0, 0, 1, 1]
    # M3 = a00(b01-b11): U_M3 = [1, 0, 0, 0]
    # M4 = a11(b10-b00): U_M4 = [0, 0, 0, 1]
    # M5 = (a00+a01)b11: U_M5 = [1, 1, 0, 0]
    # M6 = (a10-a00)(b00+b01): U_M6 = [-1, 0, 1, 0]
    # M7 = (a01-a11)(b10+b11): U_M7 = [0, 1, 0, -1]
    
    U = [
        [1.0,  0.0,  0.0,  0.0],   # M1
        [0.0,  0.0,  1.0,  1.0],   # M2
        [1.0,  0.0,  0.0,  0.0],   # M3
        [0.0,  0.0,  0.0,  1.0],   # M4
        [1.0,  1.0,  0.0,  0.0],   # M5
        [-1.0, 0.0,  1.0,  0.0],   # M6
        [0.0,  1.0,  0.0, -1.0],   # M7
    ]
    # Transpose: U is (4 rows, 7 cols) -> we want U[i, r] where i indexes the 4 entries of A
    U_T = [[U[r][i] for r in range(7)] for i in range(4)]
    
    # V: 2x2 matrix -> 7 values
    # Flattened B: [b00, b01, b10, b11]
    # M1: V = [1, 0, 0, 1]
    # M2: V = [1, 0, 0, 0]
    # M3: V = [0, 1, 0, -1]
    # M4: V = [-1, 0, 1, 0]
    # M5: V = [0, 0, 0, 1]
    # M6: V = [1, 1, 0, 0]
    # M7: V = [0, 0, 1, 1]
    
    V_raw = [
        [1.0,  0.0,  0.0,  1.0],   # M1
        [1.0,  0.0,  0.0,  0.0],   # M2
        [0.0,  1.0, 0.0, -1.0],    # M3
        [-1.0, 0.0,  1.0,  0.0],   # M4
        [0.0,  0.0,  0.0,  1.0],   # M5
        [1.0,  1.0,  0.0,  0.0],   # M6
        [0.0,  0.0,  1.0,  1.0],   # M7
    ]
    V_T = [[V_raw[r][j] for r in range(7)] for j in range(4)]
    
    # W: 7 values -> 2x2 output
    # Flattened C: [c00, c01, c10, c11]
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    W_raw = [
        [1.0,  0.0,  0.0,  1.0],   # c00: [1,0,0,1,0,0,1]
        [0.0,  0.0,  1.0,  0.0],   # c01: [0,0,1,0,1,0,0]
        [0.0,  1.0,  0.0,  0.0],   # c10: [0,1,0,1,0,0,0]
        [1.0, -1.0, 1.0,  0.0],    # c11: [1,-1,1,0,0,1,0]
    ]
    # W is (4 rows, 7 cols)
    W_T = [[W_raw[i][r] for r in range(7)] for i in range(4)]
    
    return U_T, V_T, W_T

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    The tensor decomposition says: C[i,k] = sum_r U[i*m + ..., r] * prod_r * V[..., r]
    More precisely, for each r, we have rank-1 term:
    C[i,k] += (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Wait, the standard formulation is:
    C[i,k] = sum_r U[i, r] * V[k, r] * W[r]  -- no, that's not right either.
    
    The correct formulation: 
    C = A @ B where C[i,k] = sum_j A[i,j] B[j,k]
    
    A tensor decomposition of the matmul tensor T[i,j,k] = delta(j,j) gives:
    C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k])
    
    Wait, I need to be more careful. The standard form is:
    For each factor r, we define:
    X_r[i,j] = U[i, r] (a vector of length m for each i) -- no
    
    Let me think again. The matrix multiplication can be written as:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A bilinear form decomposition of rank R means we can write:
    C[i,k] = sum_{r=1}^{R} (sum_j U_r[i,j] A[i,j]) (sum_j V_r[j,k] B[j,k])
    
    No wait, that's not standard either. Let me look at the interface:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    So U is indexed by (i*m+j) and r.
    V is indexed by (j*p+k) and r.
    W is indexed by (k*n+i) and r.
    
    The reconstruction should be:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]?
    
    That doesn't look right dimensionally. Let me reconsider.
    
    Actually, I think the standard form for this interface is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    
    Hmm, that's a product of three things per term, which is trilinear not bilinear.
    
    Let me reconsider. The task says "exactly decompose the matrix multiplication tensor".
    The matmul tensor is T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    Actually T[i,j,k] = delta_{j,j} which is always 1? No.
    
    The tensor for matrix multiplication A(n) x B(m) -> C(p) where A is n x m, B is m x p, C is n x p:
    T[i, j, k] where i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1.
    And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    
    A rank-1 decomposition: C[i,k] = U[i,j,r] * V[j,k,r] * ... 
    
    I think the correct interpretation given the shapes is:
    U: (n*m) x rank, so U[i*m+j, r]
    V: (m*p) x rank, so V[j*p+k, r]  
    W: (p*n) x rank, so W[k*n+i, r]
    
    And the reconstruction is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    No wait, that still has a product of three factors. 
    
    Actually, I think I'm overcomplicating this. Let me re-read the schoolbook example:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k).
    
    So for a fixed (i,j,k), col = i*m*p + j*p + k.
    Then C[i,k] should get contribution A[i,j] * B[j,k].
    
    With this formula: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    For r corresponding to (i,j,k):
    W[k*n+i, r] = 1
    sum_j U[i*m+j', r] * A[i,j'] = U[i*m+j, r] * A[i,j] = 1 * A[i,j] = A[i,j]
    sum_j V[j'*p+k, r] * B[j',k] = V[j*p+k, r] * B[j,k] = 1 * B[j,k] = B[j,k]
    
    So the contribution is 1 * A[i,j] * B[j,k]. Summing over all r (i.e., all (i,j,k) triples):
    C[i,k] = sum_{i',j',k'} [if (i',k')==(i,k)] A[i,j'] * B[j',k] = sum_{j'} A[i,j'] * B[j',k]
    
    Yes! That's correct. So the formula is:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Let me verify with a simple test.
    """
    # Test with small matrices
    import numpy as np
    
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    rank = U_arr.shape[1]
    C_computed = np.zeros((n, p), dtype=float)
    
    for r in range(rank):
        # sum_j U[i*m+j, r] * A[i,j] for each i
        term_U = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(m):
                term_U[i] += U_arr[i * m + j, r] * A[i, j]
        
        # sum_j V[j*p+k, r] * B[j,k] for each k
        term_V = np.zeros(p, dtype=float)
        for k in range(p):
            for j in range(m):
                term_V[k] += V_arr[j * p + k, r] * B[j, k]
        
        # C[i,k] += W[k*n+i, r] * term_U[i] * term_V[k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k * n + i, r] * term_U[i] * term_V[k]
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 110  # Leave 10s safety margin
    
    # Start with schoolbook (rank 60)
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to construct a better decomposition using Strassen-based block approach
    # Strategy: Use block decomposition where each 2x2 block pair uses Strassen (rank 7 per 2x2x2 block)
    # For 3x4x5: pad to 4x4x5? Or use different block sizes.
    
    # Actually, let's try a direct approach: build a decomposition for the 3x4x5 matmul
    # by combining smaller block decompositions.
    
    # For now, just return schoolbook as a safe fallback.
    # The Strassen approach requires careful construction.
    
    # Let's verify the schoolbook works
    if not _check_time(deadline):
        valid = _verify_decomposition(best['U'], best['V'], best['W'], n, m, p)
        if valid:
            return best
    
    # Fallback: return schoolbook anyway
    return best
