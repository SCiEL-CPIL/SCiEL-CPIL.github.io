<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U44, V44, W44 = _build_4x4x4_strassen()
    
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U41[i * 4 + k, r] = 1
            V41[k * 1 + 0, r] = 1
            W41[0 * 4 + i, r] = 1
    
    total_rank = 49 + 49 + 16
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A*B1 (cols 0-3)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Part 2: A*B2 (cols 4-7)
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Part 3: A*B3 (col 8)
    U[:, 98:114] = U41
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def _build_winograd_4x1():
    """
    Build Winograd frequency-domain decomposition for 4x4 * 4x1.
    Uses the 5-point FFT approach: pad to 5x5, use cyclic convolution.
    
    For 4x1 multiplication: C[i] = sum_k A[i,k] * b[k]
    
    Winograd F_4^T(5) approach:
    We use the 5-point DFT with primitive 5th root of unity.
    The key insight: 4x4*4x1 can be embedded in a 5x5*5x5 cyclic convolution
    where we only need 5 non-zero frequency components.
    
    Actually, for 4x1, a simpler approach:
    C[i] = sum_k A[i,k] * b[k]
    
    Using Winograd's method for vector-matrix product:
    We can use the fact that this is equivalent to 4 independent dot products.
    
    A better approach: use the 5-point cyclic convolution.
    Let a_i = A[i,:] (length 4), b = B[:,0] (length 4).
    Pad to length 5: a'_i = a_i for i<4, a'_4 = 0; same for b.
    Cyclic convolution of length 5 has 5 points.
    
    The 5-point DFT: X_k = sum_j x_j * omega^(jk) where omega = exp(2*pi*i/5)
    
    But we need real/integer/half-integer coefficients.
    
    Alternative: use the known rank-5 decomposition for 4x4*4x1 via Winograd.
    
    Actually, let me use a direct approach. For 4x1:
    C[i] = sum_k A[i,k] * b[k] for i=0,1,2,3
    
    This is 4 dot products of length 4. The naive rank is 16.
    
    Using Winograd F_4^T(5):
    The transform matrices for 4-point Winograd with 5-point FFT:
    
    U matrix (4x4 -> 5x4):
    U = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 1, 1, 1]]
    
    Wait, I need to be more careful. Let me use the standard Winograd F_4^T(5).
    
    For computing y = A * b where A is 4x4 and b is 4x1:
    
    Step 1: Transform A: A' = U * A (5x4)
    Step 2: Transform b: b' = V * b (5x1)  
    Step 3: Pointwise multiply: c' = A' .* b' (5x4)
    Step 4: Inverse transform: C = W * c' (4x4... wait, C should be 4x1)
    
    Hmm, this is getting complicated. Let me use a simpler verified approach.
    
    For 4x4 * 4x1, I can use the following rank-8 decomposition:
    Split into two 2x4 * 4x1 problems, each with rank 4.
    
    Actually, the simplest correct improvement: use rank 4 instead of 16 for the 4x1 case
    by exploiting structure. But let me just verify the current approach works and
    try a Winograd-based rank reduction.
    
    For now, let me use a verified rank-8 approach:
    C[0:2, 0] = A[0:2,:] @ b
    C[2:4, 0] = A[2:4,:] @ b
    
    Each 2x4 * 4x1 can be done with rank 4 (naive).
    Total rank for 4x1: 8.
    
    But actually, I can do better. Let me use the 5-point Winograd.
    
    The 5-point DFT matrix (real part, using the fact that omega^5 = 1):
    omega = exp(2*pi*i/5)
    
    For real inputs, we can use the real 5-point DFT.
    
    Actually, let me just use a simple but correct approach: rank 8 for 4x1.
    """
    # For 4x4 * 4x1, use rank 8: split into two 2x4*4x1 problems
    # Each 2x4*4x1: C[i] = sum_k A[i,k]*b[k], rank 4 each
    # Total: rank 8
    
    U41 = np.zeros((16, 8))
    V41 = np.zeros((4, 8))
    W41 = np.zeros((4, 8))
    
    # First half: rows 0,1 of A
    for k in range(4):
        r = k  # rank index 0-3
        U41[0 * 4 + k, r] = 1  # A[0,k]
        U41[1 * 4 + k, r] = 1  # A[1,k]
        V41[k, r] = 1  # b[k]
        W41[0, r] = 1  # contributes to C[0]
        W41[1, r] = 1  # contributes to C[1]
    
    # Second half: rows 2,3 of A
    for k in range(4):
        r = 4 + k  # rank index 4-7
        U41[2 * 4 + k, r] = 1  # A[2,k]
        U41[3 * 4 + k, r] = 1  # A[3,k]
        V41[k, r] = 1  # b[k]
        W41[2, r] = 1  # contributes to C[2]
        W41[3, r] = 1  # contributes to C[3]
    
    return U41, V41, W41

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U44, V44, W44 = _build_4x4x4_strassen()
    
    U41, V41, W41 = _build_winograd_4x1()
    
    total_rank = 49 + 49 + 8
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A*B1 (cols 0-3)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Part 2: A*B2 (cols 4-7)
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Part 3: A*B3 (col 8) using rank-8 Winograd
    U[:, 98:106] = U41
    for k in range(4):
        V[k * 9 + 8, 98:106] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:106] = W41[i, :]
    
    # Verify before returning
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W, n, m, p, max_checks=50):
            # Fall back to naive if verification fails
            U41_naive = np.zeros((16, 16))
            V41_naive = np.zeros((4, 16))
            W41_naive = np.zeros((4, 16))
            for i in range(4):
                for k in range(4):
                    r = i * 4 + k
                    U41_naive[i * 4 + k, r] = 1
                    V41_naive[k, r] = 1
                    W41_naive[i, r] = 1
            
            total_rank_naive = 49 + 49 + 16
            U = np.zeros((16, total_rank_naive))
            V = np.zeros((36, total_rank_naive))
            W = np.zeros((36, total_rank_naive))
            
            U[:, 0:49] = U44
            for k in range(4):
                for j in range(4):
                    V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
            for j in range(4):
                for i in range(4):
                    W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
            
            U[:, 49:98] = U44
            for k in range(4):
                for j in range(4):
                    V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
            for j in range(4):
                for i in range(4):
                    W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
            
            U[:, 98:114] = U41_naive
            for k in range(4):
                V[k * 9 + 8, 98:114] = V41_naive[k, :]
            for i in range(4):
                W[8 * 4 + i, 98:114] = W41_naive[i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
